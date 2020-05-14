#include "mainwindow.h"
#include <QApplication>

#include <unistd.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.026f; //meters
const float arucoSquareDimension = 0.033f; //meters
const Size chessBoardDimensions = Size(9,6);
const bool live = true; //camera=TRUE OR video=FALSE




void createKnownBoardPosition(Size boardSize, float squareEdgeLenght, vector<Point3f>& corners)
{
    for(int i=0; i < boardSize.height; i++)
    {
        for(int j=0; j < boardSize.width; j++)
        {
            corners.push_back(Point3f(j * squareEdgeLenght, i * squareEdgeLenght, 0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
    for(vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
    {
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

        if(found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if(showResults)
        {
            drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
            imshow("Looking for Corners", *iter);
            waitKey(0);
        }
    }

}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distortionCoefficients)
{
    vector<vector<Point2f>> checkerboardImagesSpacePoints;
    getChessboardCorners(calibrationImages, checkerboardImagesSpacePoints, false);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);

    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImagesSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distortionCoefficients = Mat::zeros(8, 1, CV_64F);

    calibrateCamera(worldSpaceCornerPoints, checkerboardImagesSpacePoints, boardSize, cameraMatrix, distortionCoefficients, rVectors, tVectors);


}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distorsionCoefficients)
{
    ofstream outStream(name);

    if(outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for(int r=0; r < rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distorsionCoefficients.rows;
        columns = distorsionCoefficients.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for(int r=0; r < rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double value = distorsionCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;
     }
   return false;
}

vector<Point3f> getCornersInCameraWorld(double side, Vec3d rvec, Vec3d tvec){

     double half_side = side/2;


     // compute rot_mat
     Mat rot_mat;
     Rodrigues(rvec, rot_mat);

     // transpose of rot_mat for easy columns extraction
     Mat rot_mat_t = rot_mat.t();

     // the two E-O and F-O vectors
     double * tmp = rot_mat_t.ptr<double>(0);
     Point3f camWorldE(tmp[0]*half_side,
                       tmp[1]*half_side,
                       tmp[2]*half_side);

     tmp = rot_mat_t.ptr<double>(1);
     Point3f camWorldF(tmp[0]*half_side,
                       tmp[1]*half_side,
                       tmp[2]*half_side);

     // convert tvec to point
     Point3f tvec_3f(tvec[0], tvec[1], tvec[2]);

     // return vector:
     vector<Point3f> ret(4,tvec_3f);

     ret[0] +=  camWorldE + camWorldF;
     ret[1] += -camWorldE + camWorldF;
     ret[2] += -camWorldE - camWorldF;
     ret[3] +=  camWorldE - camWorldF;

     return ret;
}

// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3d &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;

    return R;

}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());

    return  norm(I, shouldBeIdentity) < 1e-6;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat &R)
{
    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }

    return Vec3f(x, y, z);
}

vector<float> checkDifferentMarkers(vector<int> markerIds)
{


    vector<float> ret;

    for(int c=0; c < markerIds.size(); c++)
    {
        if(markerIds[c] == 0 || markerIds[c] == 1)
            ret.push_back(0.080f);
        else if (markerIds[c] == 2 || markerIds[c] == 3)
            ret.push_back(0.100f);
        else if (markerIds[c] == 4 || markerIds[c] == 5)
            ret.push_back(0.100f);
        else if (markerIds[c] == 6 || markerIds[c] == 7)
            ret.push_back(0.080f);
        else if (markerIds[c] == 8 || markerIds[c] == 9)
            ret.push_back(0.070f);
        else if (markerIds[c] == 10 || markerIds[c] == 11)
            ret.push_back(0.060f);
        else if (markerIds[c] == 12 || markerIds[c] == 13)
            ret.push_back(0.050f);
        else if (markerIds[c] == 14 || markerIds[c] == 15 || markerIds[c] == 16 || markerIds[c] == 17)
            ret.push_back(0.040f);
        else if (markerIds[c] == 18 || markerIds[c] == 19 || markerIds[c] == 20 || markerIds[c] == 21)
            ret.push_back(0.030f);
        else if (markerIds[c] == 22 || markerIds[c] == 23 || markerIds[c] == 24 || markerIds[c] == 25)
            ret.push_back(0.020f);
        else
            ret.push_back(0.000f);

    }

    return ret;

}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distortionCoefficients)
{
    Mat frame;
    vector<Point3f> pos;

    vector<int> markerIds;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    aruco::DetectorParameters parameters;
    parameters.polygonalApproxAccuracyRate = 0.05; //0.05 default

    Ptr < aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);

    //Camera = 0
    //PathToFile = "/home/joao/Videos/Webcam/Aruco/close.webm"
    VideoCapture vid(0);

    if(!vid.isOpened())
    {
        return -1;
    }

    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    vector<Vec3d> rotationVectors, translationVectors;
    vector<Vec3d> rotationVectorsDegrees;
    Vec3f angles;

    long frameCounter = 0;

    time_t timeBegin = time(0);
    int tick = 0;

    while(true)
    {
        if(!vid.read(frame))
            break;

        frameCounter++;

        std::time_t timeNow = std::time(0) - timeBegin;

        if (timeNow - tick >= 1)
        {
            tick++;
            cout << "Frames per second: " << frameCounter << endl;
            frameCounter = 0;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds); // Verificar se existe flags

        vector<float> arucoSquareDimensions = checkDifferentMarkers(markerIds);

        for(int i=0; i < markerIds.size(); i++)
        {
            aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimensions[i], cameraMatrix, distortionCoefficients, rotationVectors, translationVectors);

            //aruco::drawAxis(frame, cameraMatrix, distortionCoefficients, rotationVectors[i], translationVectors[i], 0.05f);
            aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            // Calc camera pose
            Mat R, rotationMatrix;
            Rodrigues(rotationVectors[i], R);
            Mat cameraPose = -R.t() * (Mat)translationVectors[i];

            double x = cameraPose.at<double>(0,0);
            double y = cameraPose.at<double>(0,1);
            double z = cameraPose.at<double>(0,2);
            double distance = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));

            //rotationMatrix = eulerAnglesToRotationMatrix(rotationVectors[i]);
            angles = rotationMatrixToEulerAngles(R);

            cout << fixed << setprecision(0); // Casas decimais
            cout << "MarkerId: " << markerIds[i] << " " << endl;
            //cout << " Distance: " << distance * 1000 << endl;
            //cout << "Angles [X(red), Y(green), Z(blue)]: " << angles * 180 / M_PI << endl;

        }



        imshow("Webcam", frame);
        if(waitKey(30) >= 0) break;

        //sleep(1);
    }


    return 1;
}


void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distortionCoefficients)
{
    Mat frame;
    Mat drawToFrame;

    vector<Mat> savedImages;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(0);

    if(!vid.isOpened())
    {
        return;
    }

    int framesPerSecond = 15;

    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
            break;

        vector<Vec2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, chessBoardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessBoardDimensions, foundPoints, found);

        if(found)
            imshow("Webcam", drawToFrame);
        else
            imshow("Webcam", frame);

        char character = waitKey(1000 / framesPerSecond);

        switch(character)
        {
        case ' ':
            //saving image
            if(found)
            {
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
                cout << savedImages.size() << endl;
            }
            break;
        case 8:
            //start calibration (BACKSPACE key)
            if(savedImages.size() > 15)
            {
                cameraCalibration(savedImages, chessBoardDimensions, calibrationSquareDimension, cameraMatrix, distortionCoefficients);
                bool result = saveCameraCalibration("camera_calibration_parameters", cameraMatrix, distortionCoefficients);
                if(result)
                    cout << "1" << endl;
                else
                    cout << "0" << endl;
            }

            break;
        case 27:
            //exit
            return;
        }
    }

    return;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distortionCoefficients)
{
    ifstream inStream(name);
    if(inStream)
    {
        uint16_t rows;
        uint16_t columns;

        inStream >> rows;
        inStream >> columns;

        cameraMatrix = Mat(Size(columns, rows), CV_64F);

        for(int r=0; r < rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r, c) = read;
                cout << cameraMatrix.at<double>(r, c) << "\n";
            }
        }

        //Distorsion Coefficients
        inStream >> rows;
        inStream >> columns;

        distortionCoefficients = Mat::zeros(rows, columns, CV_64F);

        for(int r=0; r < rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                distortionCoefficients.at<double>(r, c) = read;
                cout << distortionCoefficients.at<double>(r, c) << "\n";
            }
        }
        inStream.close();
        return true;

    }

    return false;
}

void openCameraFunction()
{
    cv::VideoCapture cam;

    if (!cam.open(0))
        cout << "Problem connecting to cam " << std::endl;
    else
        cout << "Successfuly connected to camera " << std::endl;

    long frameCounter = 0;

    time_t timeBegin = time(0);
    int tick = 0;

    Mat frame;

    while(1)
    {
        cam.read(frame);

        cv::imshow("Img", frame);
        cv::waitKey(1);

        frameCounter++;

        std::time_t timeNow = std::time(0) - timeBegin;

        if (timeNow - tick >= 1)
        {
            tick++;
            cout << "Frames per second: " << frameCounter << endl;
            frameCounter = 0;
        }
    }

    return;
}

int main(int argc, char **argv)
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();


    Mat cameraMatrix = Mat::eye(3,3, CV_64F);

    Mat distortionCoefficients;
    cout << "1";

    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);

    //cameraCalibrationProcess(cameraMatrix, distortionCoefficients);
    //loadCameraCalibration("camera_calibration_parameters", cameraMatrix, distortionCoefficients);
    //startWebcamMonitoring(cameraMatrix, distortionCoefficients);
    //openCameraFunction();

    return a.exec();
}
