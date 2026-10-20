"""PhysX articulation exclusions, without unrelated planning SRDF exclusions."""


def adjacent_body_pairs(urdf, represented_bodies):
    """Return adjacent exported rigid bodies; some fixed joints retain both bodies."""
    groups={'base_link':'base_link'}
    pending=list(urdf.findall('joint'));adjacent=set()
    while pending:
        ready=[]
        for joint in pending:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            if parent not in groups:continue
            separate=child in represented_bodies
            groups[child]=child if separate else groups[parent]
            if separate:adjacent.add(frozenset((groups[parent],groups[child])))
            ready.append(joint)
        if not ready:raise ValueError('Disconnected training URDF')
        pending=[j for j in pending if j not in ready]
    return {pair for pair in adjacent if len(pair)==2 and all(name in represented_bodies for name in pair)}


def planner_body_pairs(urdf, represented_bodies, model, srdf):
    """Keep historical profiles reproducible; opt new profiles into USD adjacency."""
    if model=='usd_adjacent':return adjacent_body_pairs(urdf,represented_bodies)
    if model!='legacy_srdf':raise ValueError('Unknown planner self-collision model')
    groups={'base_link':'base_link'};pending=list(urdf.findall('joint'));pairs=set()
    while pending:
        ready=[]
        for joint in pending:
            parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
            if parent not in groups:continue
            movable=joint.get('type')!='fixed';groups[child]=child if movable else groups[parent]
            if movable:pairs.add(frozenset((groups[parent],groups[child])))
            ready.append(joint)
        if not ready:raise ValueError('Disconnected training URDF')
        pending=[j for j in pending if j not in ready]
    pairs.add(frozenset(('right_fr3_leftfinger','right_fr3_rightfinger')))
    for element in srdf.findall('disable_self_collision'):
        pairs.add(frozenset((element.get('link1'),element.get('link2'))))
    return {pair for pair in pairs if len(pair)==2 and all(name in represented_bodies for name in pair)}


def planner_collision_settings(profile):
    settings=profile.get('expert_collision')
    if settings is None:return dict(self_collision_model='legacy_srdf',self_collision_buffer_m=.03,distal_collision_buffer_m=.05)
    return dict(self_collision_model=settings.get('model','usd_adjacent'),
                self_collision_buffer_m=settings.get('self_buffer_m',.03),
                distal_collision_buffer_m=settings.get('distal_buffer_m',.05))
