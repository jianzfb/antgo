from ..builder import PIPELINES
import torchvision.transforms as transforms


@PIPELINES.register_module()
class Compose(transforms.Compose):
    def __init__(self, processes):
        process_t = []
        for pn, pp in processes:
            process_t.append(getattr(transforms, pn)(**pp))

        super().__init__(process_t)
    
    def __call__(self, sample):
        image = sample['image']
        sample['image'] = super(Compose, self).__call__(image)
        return sample

