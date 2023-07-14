file_path = r'C:\Users\Omnia\Desktop\Batch 001\NH13-581\NH13-581.scn'
pro_path =r"C:\Users\Omnia\Desktop\Batch 001\tiles/"
from histolab.slide import Slide
prostate_slide = Slide(file_path,pro_path)

print(f"Slide name: {prostate_slide.name}")
print(f"Levels: {prostate_slide.levels}")
print(f"Dimensions at level 0: {prostate_slide.dimensions}")
print(f"Dimensions at level 1: {prostate_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {prostate_slide.level_dimensions(level=2)}")

print(
     "Native magnification factor:",
     prostate_slide.level_magnification_factor()
 )

print(
     "Magnification factor corresponding to level 1:",
     prostate_slide.level_magnification_factor(level=1),
 )

prostate_slide.thumbnail
prostate_slide.show()
#

from histolab.masks import TissueMask, BiggestTissueBoxMask
from histolab.slide import Slide


tissue_mask = TissueMask() # or BiggestTissueBoxMask()

img = prostate_slide.locate_mask(tissue_mask, alpha=255)
    
from histolab.tiler import GridTiler



from histolab.tiler import GridTiler

grid_tiles_extractor = GridTiler(
   tile_size=(512, 512),
   level=0,
   check_tissue=False,
   pixel_overlap=0, # default
   prefix = pro_path, # save tiles in the "grid" subdirectory of slide's processed_path
   suffix=".png" # default
)
# grid_tiles_extractor.locate_tiles(
#     slide=prostate_slide,
#     scale_factor=64,
#     alpha=64,
#     outline="#046C4C",
# )
grid_tiles_extractor.extract(prostate_slide)
