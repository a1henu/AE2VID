from utils.loading_utils import get_device
from utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, ImageWriter, UnsharpMaskFilter
from utils.timers import CudaTimer

import torch

class AdaE2VIDReconstructor():
    def __init__(self, model, adapter, options, height, width):
        self.use_gpu = options.use_gpu
        self.device = get_device(self.use_gpu)
        self.model = model.to(self.device)
        self.adapter = adapter.to(self.device)
        
        self.model.eval()
        self.adapter.eval()

        self.height = height
        self.width = width
        self.num_bins = self.model.num_bins
        
        self.prev_state = None
        
        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)
        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.image_writer = ImageWriter(options)
        self.image_display = ImageDisplay(options)
    
    def initialize(self, init_frame):
        """Initialize the reconstructor with given options and initial frame.
         init_frame: [1, 1, H, W] tensor
         """
        # print('== AdaE2VID Reconstruction == ')
        # print('Image size: {}x{}'.format(self.width, self.height))
        init_vox = init_frame.repeat(1, self.num_bins, 1, 1)
        init_vox = self.crop.pad(init_vox)
        # print(init_vox.shape)
        with torch.no_grad():
            _, prev_state = self.adapter(init_vox, self.prev_state)
            # _, prev_state = self.adapter(init_vox, None)
            self.prev_state = prev_state
    
    def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):
        with torch.no_grad():
            with CudaTimer('AdaE2VID Reconstruction'):
                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor.unsqueeze(dim=0)
                    events = events.to(self.device)

                events = self.event_preprocessor(events)
                events_cropped = self.crop.pad(events)

                with CudaTimer('Inference'):
                    recon_frame, states = self.model(events_cropped, self.prev_state)
                
                self.prev_state = states
                recon_frame = self.unsharp_mask_filter(recon_frame)
                recon_frame = self.intensity_rescaler(recon_frame)
                
                with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                    recon_frame = recon_frame[0, 0, self.crop.iy0:self.crop.iy1, self.crop.ix0:self.crop.ix1].cpu().numpy()
        
        recon_frame = self.image_filter(recon_frame)
        self.image_writer(recon_frame, event_tensor_id, stamp, events)