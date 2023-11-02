// this script to patch the roi

import static qupath.lib.gui.scripting.QPEx.*

server  = getCurrentServer()
path = server.getPath()
downsample = 1.0

name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
pathOutput = buildFilePath(PROJECT_BASE_DIR, "patches", name)
pathOutput_Tumor = buildFilePath(PROJECT_BASE_DIR, "patches", name, "Tumor")
pathOutput_Negative = buildFilePath(PROJECT_BASE_DIR, "patches", name, "Negative")
mkdirs(pathOutput)
mkdirs(pathOutput_Tumor)
mkdirs(pathOutput_Negative)


i = 1
j = 1
for (annotation in getAnnotationObjects()){
    roi = annotation.getROI()
    request = RegionRequest.createInstance(path, downsample, roi)
    writeImageRegion(server, request, pathOutput_Negative + "/tile_" + j + "_" + roi.toString() + '.tiff')
        j = j + 1
     
}

// this to make a binary mask within the anotation 

/**
 * Script to export binary masks corresponding to all annotations of an image,
 * optionally along with extracted image regions.
 *
 * Note: Pay attention to the 'downsample' value to control the export resolution!
 *
 * @author Pete Bankhead
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getAnnotationObjects()

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 4.0
def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR, 'masks')
mkdirs(pathOutput)

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as PNG
def imageExportType = 'JPG'

// Export each annotation
annotations.each {
    saveImageAndMask(pathOutput, server, it, downsample, imageExportType)
}
print 'Done!'

/**
 * Save extracted image region & mask corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImageAndMask(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    // Extract ROI & classification name
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }

    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s_%s_(%.2f,%d,%d,%d,%d)',
            server.getMetadata().getName(),
            classificationName,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = RoiTools.getShape(roi)
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())
    g2d.fill(shape)
    g2d.dispose()

    // Create filename & export
    if (imageExportType != null) {
        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
        ImageIO.write(img, imageExportType, fileImage)
    }
    // Export the mask
    def fileMask = new File(pathOutput, name + '-mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)

}

// This script would create a contour around the tissue

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

def pixelSizeMicrons = 5 // 20
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
println("sum of areas " + labelAreas*.value.sum().toString())

def sum_area = labelAreas*.value.sum()
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
