"""Subscriber-only ROS worker, started only for an explicit lab recording."""
import argparse,json,os,time
from collections import deque
from pathlib import Path
import numpy as np
from recording import RawRecorder,atomic,require_capture_ready,static_camera_info,rectification_maps


def main(config_path):
    import rclpy,cv2
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ROSImage,CameraInfo
    from message_filters import Subscriber,ApproximateTimeSynchronizer,TimeSynchronizer
    spec=json.loads(Path(config_path).read_text(encoding='utf-8'));root=Path(spec['output']);cfg=spec['calibration'];owner=os.getppid()
    cfg=require_capture_ready(cfg)
    recorder=RawRecorder(root,spec['map'],spec['map_image'],spec['roi'],cfg)
    stamp=lambda m:m.header.stamp.sec+m.header.stamp.nanosec*1e-9
    def array(msg,channels,dtype):
        dt=np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<');width=msg.width*channels*dt.itemsize
        if msg.width*msg.height>4_000_000 or msg.step<width or len(msg.data)!=msg.height*msg.step:raise ValueError('图像缓冲无效')
        rows=np.frombuffer(bytes(msg.data),np.uint8).reshape(msg.height,msg.step)[:,:width].copy()
        return rows.view(dt).reshape((msg.height,msg.width,channels) if channels>1 else (msg.height,msg.width))
    class Capture(Node):
        def __init__(self):
            super().__init__('astra_mapping_capture_'+root.name[:8]);self.info=None;self.last_pair=0;self.map_key=None;self.maps=None
            if cfg['intrinsics_source']=='camera_info':self.create_subscription(CameraInfo,cfg['info_topic'],self.camera_info,qos_profile_sensor_data)
            self.rgb=Subscriber(self,ROSImage,cfg['rgb_topic'],qos_profile=qos_profile_sensor_data)
            self.depth=Subscriber(self,ROSImage,cfg['depth_topic'],qos_profile=qos_profile_sensor_data)
            self.sync=TimeSynchronizer([self.rgb,self.depth],10) if cfg['intrinsics_source']=='static' else ApproximateTimeSynchronizer([self.rgb,self.depth],10,.05)
            self.sync.registerCallback(self.pair)
        def camera_info(self,msg):self.info=msg
        def pair(self,rgb,depth):
            if time.monotonic()-self.last_pair<.8/cfg['capture_hz']:return
            self.last_pair=time.monotonic()
            try:
                ts=stamp(rgb)
                if cfg['intrinsics_source']=='static':info=static_camera_info(cfg,rgb,depth)
                else:
                    info=self.info
                    if info is None:return recorder.reject('等待 CameraInfo 内参')
                    if (info.width,info.height)!=(rgb.width,rgb.height) or rgb.header.frame_id!=info.header.frame_id or depth.header.frame_id!=info.header.frame_id:return recorder.reject('内参/深度/彩色光学坐标系或分辨率不一致')
                if rgb.encoding not in ('rgb8','bgr8'):return recorder.reject('不支持的彩色编码')
                color=array(rgb,3,np.uint8)
                if rgb.encoding=='bgr8':color=color[:,:,::-1].copy()
                if depth.encoding=='16UC1':d=array(depth,1,np.uint16).astype(np.float32)*.001
                elif depth.encoding=='32FC1':d=array(depth,1,np.float32)
                else:return recorder.reject('不支持的深度编码')
                K=np.array(info.k).reshape(3,3);D=np.array(info.d)
                if np.any(np.abs(D)>1e-10):
                    key=(tuple(K.ravel()),tuple(D),info.distortion_model,rgb.width,rgb.height)
                    if key!=self.map_key:
                        self.maps=rectification_maps(K,D,info.distortion_model,(rgb.width,rgb.height));self.map_key=key
                    mx,my=self.maps
                    color=cv2.remap(color,mx,my,cv2.INTER_LINEAR);d=cv2.remap(d,mx,my,cv2.INTER_NEAREST)
                recorder.add(color,d,K,ts)
            except Exception as e:recorder.reject(str(e))
    rclpy.init();node=Capture();last_check=0
    try:
        while rclpy.ok() and recorder.state!='STOPPED':
            if (root.parent/(root.name+'.stop')).exists() or os.getppid()!=owner:break
            limit=cfg['max_duration_s']
            if limit and time.monotonic()-recorder.started>limit:recorder.reason='达到配置的采集时长上限';break
            rclpy.spin_once(node,timeout_sec=.2)
            if time.monotonic()-last_check>3:
                from urllib.request import urlopen
                try:
                    now=json.load(urlopen(spec['nav_url']+'/viz/api/map/metadata',timeout=2));expected=spec['map']['source']['signature']
                    if any(now.get(k)!=v for k,v in expected.items()):recorder.reason='导航地图已切换，停止采集';break
                except Exception:recorder.reason='无法核对导航地图版本，停止采集';break
                last_check=time.monotonic();recorder.save_status()
    finally:
        recorder.stop(recorder.reason if recorder.reason!='已保存原始 RGB-D；位姿将在停止后估计' else '采集完成');node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:main(a.config)
    except Exception as e:
        spec=json.loads(Path(a.config).read_text(encoding='utf-8'));root=Path(spec['output']);root.mkdir(exist_ok=True)
        atomic(root/'capture_status.json',dict(state='FAILED',reason=str(e),frames=0,updated_at=time.time()))
        raise
