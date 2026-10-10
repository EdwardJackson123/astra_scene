"""Separate artifact readiness from unverified navigation acceptance."""
def navigation_assessment(report):
    blockers=[]
    if report.get('alignment_accepted') is False: blockers.append('ALIGNMENT_QUALITY_FAILED')
    if report.get('metric_scale_conflict',True): blockers.append('METRIC_SCALE_CONFLICT_OR_UNKNOWN')
    if report.get('absolute_metric_accuracy_verified') is not True: blockers.append('METRIC_SCALE_NOT_VERIFIED')
    if report.get('robot_clearance_verified') is not True: blockers.append('ROBOT_CLEARANCE_NOT_VERIFIED')
    if report.get('independent_revisit_verified') is not True: blockers.append('INDEPENDENT_REVISIT_NOT_VERIFIED')
    if report.get('outside_allowed_unchanged') is not True: blockers.append('UPDATE_SCOPE_NOT_VERIFIED')
    if report.get('cleared_with_holdout_obstacle',0)>0: blockers.append('HELD_OUT_OBSTACLE_CONFLICT')
    # No automatic navigation acceptance authority exists in this service.
    return dict(state='BLOCKED' if blockers else 'PENDING_REVIEW',blockers=blockers,
                accepted=False,published=False)


def status_summary(state):
    update=state.get('map_update',{})
    return dict(reconstruction='READY' if state.get('reconstruction_ready') else 'NOT_READY',
        fusion=update.get('state','NOT_STARTED'),
        grid='CANDIDATE_READY' if update.get('navigation_candidate_ready') else 'NOT_READY',
        grid_backend=update.get('grid_backend','UNKNOWN'),
        navigation=update.get('navigation_assessment',dict(state='NOT_VERIFIED',accepted=False)),
        published=bool(update.get('published',False)))
