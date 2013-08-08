#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


int main(int argc, char **argv) {

    // Face Detect Variables
    const char *cascade_name = "/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
    CvHaarClassifierCascade *cascade = 0;
    CvMemStorage *storage = 0;
    CvSeq *faces;
    static CvScalar colors[] = {{{255, 255, 255}}};
    
    // Initialize Text
    int font_face[] = {CV_FONT_HERSHEY_SIMPLEX};
    char *coordinate;
    CvFont font[1];
    cvInitFont (&font[0], font_face[0], 0.3, 0.3);
    
    // Initialize Camera
    IplImage *frame = 0, *src_gray = 0;
    CvCapture *capture = 0;
    capture = cvCaptureFromCAM(1);
    double w = 400, h = 300;
    
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_WIDTH, w);
    cvSetCaptureProperty (capture, CV_CAP_PROP_FRAME_HEIGHT, h);
    cvNamedWindow ("Capture", CV_WINDOW_AUTOSIZE);

    
    // Image Capture
    while(1) {
        frame = cvQueryFrame (capture);

        // Faces Detect
        src_gray = cvCreateImage (cvGetSize (frame), IPL_DEPTH_8U, 1);
        
        cascade = (CvHaarClassifierCascade *) cvLoad (cascade_name, 0, 0, 0);
        
        storage = cvCreateMemStorage (0);
        cvClearMemStorage (storage);
        cvCvtColor (frame, src_gray, CV_BGR2GRAY);
        cvEqualizeHist (src_gray, src_gray);
        
        faces = cvHaarDetectObjects(src_gray, cascade, storage, 1.11, 4, 0, cvSize(40, 40), cvSize(600, 600));

        // Marking Faces
        for (int i = 0; i < (faces ? faces->total : 0); i++) {
            CvRect *r = (CvRect *) cvGetSeqElem (faces, i);
            CvPoint center;
            int radius;
            center.x = cvRound (r->x + r->width * 0.5);
            center.y = cvRound (r->y + r->height * 0.5);
            radius = cvRound ((r->width + r->height) * 0.25);
            cvCircle (frame, center, radius, colors[0], 3, 8, 0);
            
            sprintf(coordinate, "%d, %d", center.x, center.y);
            cvPutText (frame, coordinate, center, &font[0], colors[0]);
        }
       
        cvShowImage ("Capture", frame);
        cvSaveImage("/Users/yamac/desktop/cap.png", frame);
        int c = cvWaitKey (2);
        if (c == '\x1b')
            break;
    }
    
    cvReleaseCapture (&capture);
    cvReleaseImage (&src_gray);
    cvReleaseMemStorage (&storage);
    cvDestroyWindow ("Capture");
    
    return 0;
}