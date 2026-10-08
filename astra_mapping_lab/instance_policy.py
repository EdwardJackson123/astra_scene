"""Versioned fusion selection; old contexts retain their original behavior."""
import json,os
from pathlib import Path

DEFAULT_FUSION_POLICY='bounded_context_v2'
LEGACY_FUSION_POLICY='legacy_geometry'
FUSION_POLICIES=(LEGACY_FUSION_POLICY,'bounded_context_v1',DEFAULT_FUSION_POLICY)


def enabled():
    return policy()=='reviewed_instance_v1'


def policy():
    context=os.environ.get('ASTRA_UPDATE_CONTEXT')
    return json.loads(Path(context).read_text()).get('fusion_policy','') if context else ''
