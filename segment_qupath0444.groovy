// foreground extraction for brightfield histology 
// adapted from
// Bug, Daniel, Friedrich Feuerhake, und Dorit Merhof.
// Foreground Extraction for Histopathological Whole Slide Imaging.
// In Bildverarbeitung fuer die Medizin 2015, 419-424. Springer, 2015.
// http://link.springer.com/chapter/10.1007/978-3-662-46224-9_72.


import static qupath.lib.gui.scripting.QPEx.*



import ij.*
import ij.measure.Calibration
import ij.plugin.filter.ThresholdToSelection
import ij.process.ByteProcessor
import ij.process.ImageProcessor

import qupath.imagej.tools.IJTools
import qupath.lib.roi.*
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import qupath.opencv.tools.OpenCVTools

import qupath.lib.regions.*
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ServerTools
import qupath.lib.objects.PathObject
import qupath.lib.plugins.parameters.ParameterList

import static org.bytedeco.opencv.global.opencv_core.bitwise_not
import static org.bytedeco.opencv.global.opencv_core.subtract
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_NONE
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY
import static org.bytedeco.opencv.global.opencv_imgproc.Canny
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_DILATE
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_ERODE
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_CLOSE
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_OPEN
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_TOPHAT
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_ELLIPSE
import static org.bytedeco.opencv.global.opencv_imgproc.MORPH_RECT
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_TREE
import static org.bytedeco.opencv.global.opencv_imgproc.RETR_EXTERNAL
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_OTSU
import static org.bytedeco.opencv.global.opencv_imgproc.createCLAHE
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor
import static org.bytedeco.opencv.global.opencv_imgproc.findContours
import static org.bytedeco.opencv.global.opencv_imgproc.fillPoly
import static org.bytedeco.opencv.global.opencv_imgproc.getStructuringElement
import static org.bytedeco.opencv.global.opencv_imgproc.morphologyEx
import static org.bytedeco.opencv.global.opencv_imgproc.threshold
import static org.bytedeco.opencv.global.opencv_imgproc.connectedComponentsWithStats

import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.MatVector
import org.bytedeco.opencv.opencv_core.Scalar
import org.bytedeco.opencv.opencv_core.Size
import org.bytedeco.opencv.opencv_imgproc.CLAHE

def MAX_KERNEL_SIZE = 8 // MIN_PARTICLE_AREA = 400

def pixelSizeMicrons = 20 // 20, 5
def downsample = 5
def server = getCurrentServer()
def path = server.getPath()

def cal_init = server.getPixelCalibration()
if (cal_init.hasPixelSizeMicrons()) {
	downsample = pixelSizeMicrons / cal_init.getAveragedPixelSizeMicrons()
}

println("downsample " + downsample.toString())
						
def request = RegionRequest.createInstance(path, downsample, 0, 0, server.getWidth(), server.getHeight())
def bufimg = server.readBufferedImage(request)

Mat tissueMask = OpenCVTools.imageToMat(bufimg)

// extract structure map
cvtColor(tissueMask, tissueMask, COLOR_BGR2GRAY)
bitwise_not(tissueMask, tissueMask)

CLAHE clahe = createCLAHE()
clahe.setClipLimit(3.0)
clahe.setTilesGridSize(new Size(32, 32))
clahe.apply(tissueMask, tissueMask)

Canny(tissueMask, tissueMask, 60, 180)

GaussianBlur(tissueMask, tissueMask, new Size(1, 1), 2)
threshold(tissueMask, tissueMask, 0, 255, THRESH_BINARY + THRESH_OTSU)

// remove horizontal and vertical lines from partial scans (CZI)	TODO make optional
def hline_length = 40
def vline_length = 40

if(tissueMask.cols() < hline_length)
	hline_length = tissueMask.cols()
if(tissueMask.rows() < vline_length)
	vline_length = tissueMask.rows()
		
Mat hor_kernel = getStructuringElement(MORPH_RECT, new Size(hline_length, 1))
Mat ver_kernel = getStructuringElement(MORPH_RECT, new Size(1, vline_length))
morphologyEx(tissueMask, tissueMask, MORPH_TOPHAT, hor_kernel)
morphologyEx(tissueMask, tissueMask, MORPH_TOPHAT, ver_kernel)

// clean up mask
Mat kernel7 = getStructuringElement(MORPH_ELLIPSE, new Size(7, 7))
morphologyEx(tissueMask, tissueMask, MORPH_CLOSE, kernel7)

// detect outer contour of tissue:
MatVector contours = new MatVector()
Mat hierarchy = new Mat()
findContours(tissueMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
fillPoly(tissueMask, contours, new Scalar(255))

// get area info of tissue fragments
Mat labels = new Mat()
Mat stats = new Mat()
Mat centroids = new Mat()
connectedComponentsWithStats(tissueMask, labels, stats, centroids, 8, 4) // CV_32S   4
List<Integer> labelAreas = new ArrayList<>()
for (int i=1; i < stats.rows(); i++) {
	labelAreas.add(stats.getIntBuffer().get(i*5+4)) // row + n_cols + CC_STAT_AREA = 4
}

println("max area " + labelAreas.max().toString())
println("mean area " + labelAreas.average() )
println("particle areas " + labelAreas.sort().reverse().toString())
println("sum of areas " + labelAreas.sum().toString())


def sum_area = labelAreas.sum()
labelAreas = labelAreas.sort().reverse()

def sum = 0
def j = 0
for(int i=0; i < labelAreas.size()-1; i++) {
	sum += labelAreas[i]
	if(sum > 0.85 * sum_area) {
		j = i + 1
		break
	}
}
println("sum 85% " + sum.toString())
println("next particle area " + labelAreas[j].toString())

def largeKernelSize = (int)Math.ceil(Math.sqrt((double)labelAreas[j]))
if(largeKernelSize > MAX_KERNEL_SIZE) {
	largeKernelSize = MAX_KERNEL_SIZE
}

println("kernel size to remove small tissue fragments " + largeKernelSize.toString())

// remove small tissue fragments and expand mask a little bit
Mat largeKernel = getStructuringElement(MORPH_ELLIPSE, new Size(largeKernelSize, largeKernelSize))
morphologyEx(tissueMask, tissueMask, MORPH_OPEN, largeKernel)
morphologyEx(tissueMask, tissueMask, MORPH_DILATE, kernel7)

ImagePlus impNew = OpenCVTools.matToImagePlus(tissueMask, "mask")

// clean-up
tissueMask.release()
kernel7.release()
largeKernel.release()
hor_kernel.release()
ver_kernel.release()
labels.release()
centroids.release()
stats.release()

def bp = impNew.getProcessor().convertToByteProcessor()
def cal = impNew.getCalibration()

// Create a classification, if necessary
def classificationString = "Positive"
def pathClass = null
if (classificationString != 'None')
	pathClass = PathClassFactory.getPathClass(classificationString)
	
// To create the ROI, travel into ImageJ
bp.setThreshold(127.5, Double.MAX_VALUE, ImageProcessor.NO_LUT_UPDATE)
def roiIJ = new ThresholdToSelection().convert(bp)

int z = 0
int t = 0
def plane = ImagePlane.getPlane(z, t)

// Convert ImageJ ROI to a QuPath ROI
// This assumes we have a single 2D image (no z-stack, time series)
def roi = IJTools.convertToROI(roiIJ, cal, downsample, plane)

def annotation = new PathAnnotationObject(roi, pathClass)
addObject(annotation)

println("done")
