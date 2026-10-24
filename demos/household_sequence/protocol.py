"""Handover checks for the two stages of the continuous demonstration."""
def verify_bowl_delivery(events):
    if not isinstance(events, list):
        raise ValueError('Expected the bowl task event list')
    if any(event.get('event') == 'failed' for event in events):
        raise ValueError('Bowl task recorded a failure')
    delivered = None
    for event in events:
        if event.get('event') == 'transport_result':
            delivered = event.get('passed') is True
        if event.get('event') == 'completed':
            if delivered is not True:
                raise ValueError('Completion must follow a successful physical placement check')
            return
    raise ValueError('Bowl delivery is not complete')
