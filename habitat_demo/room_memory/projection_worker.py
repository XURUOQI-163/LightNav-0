"""Select one backend-specific projection pipeline; geometry/fusion stay shared."""


def SemanticProjectionWorker(service, config=None):
    if service.semantic_backend == 'groundingdino':
        from .groundingdino_projection import GroundingDINOProjectionWorker
        return GroundingDINOProjectionWorker(service, config)
    from .clip_projection import CLIPProjectionWorker
    return CLIPProjectionWorker(service, config)
