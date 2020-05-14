#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <unistd.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include <QDir>
#include <QFile>
#include <QDebug>

#include <algorithm>
#include <vector>

typedef vector<int> IntContainer;
typedef IntContainer::iterator IntIterator;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    timer = new QTimer(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openCameraFunction()
{
    cap.open(0);

    if (!cap.open(0))
        cout << "Problem connecting to cam " << std::endl;
    else
        cout << "Successfuly connected to camera " << std::endl;

    connect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
    timer->start(20);

    cv::waitKey(1);


    return;
}

void MainWindow::on_openCamera_clicked()
{
    recordFrames = false;
    loadCameraCalibration("camera_calibration_parameters", cameraMatrix, distortionCoefficients);
    startWebcamMonitoring(cameraMatrix, distortionCoefficients);
}

void MainWindow::on_closeCamera_clicked()
{
    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
    cap.release();

    Mat image = Mat::zeros(frame.size(),CV_8UC3);

    qt_image = QImage((const unsigned char*) (image.data), image.cols, image.rows, QImage::Format_RGB888);

    ui->cameraLabel->setPixmap(QPixmap::fromImage(qt_image));

    ui->cameraLabel->resize(ui->cameraLabel->pixmap()->size());

    //cout << "camera is closed" << endl;
}

void MainWindow::update_window(Mat frame)
{
    //cap >> frame;

    cvtColor(frame, frame, CV_BGR2RGB);

    qt_image = QImage((const unsigned char*) (frame.data), frame.cols, frame.rows, QImage::Format_RGB888);

    ui->cameraLabel->setPixmap(QPixmap::fromImage(qt_image));

    ui->cameraLabel->resize(ui->cameraLabel->pixmap()->size());
}

bool MainWindow::loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distortionCoefficients)
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

int MainWindow::startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distortionCoefficients)
{
    Mat frame;
    vector<Point3f> pos;
    int nFrames = 0;

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

    vector<Vec3d> rotationVectors, translationVectors;
    vector<Vec3d> rotationVectorsDegrees;
    Vec3f angles;


    while(true)
    {
        if(!vid.read(frame))
            break;

        nFrames++;

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds); // Verificar se existe flags

        vector<float> arucoSquareDimensions = checkDifferentMarkers(markerIds);

        vector<float> markerDistances;

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

            markerDistances.push_back(distance);

            //rotationMatrix = eulerAnglesToRotationMatrix(rotationVectors[i]);
            angles = rotationMatrixToEulerAngles(R);

            //cout << fixed << setprecision(0); // Casas decimais
            //cout << "MarkerId: " << markerIds[i] << " ";
            //cout << " Distance: " << distance * 1000 << endl;
            //cout << "Angles [X(red), Y(green), Z(blue)]: " << angles * 180 / M_PI << endl;

        }

        if(recordFrames && nFrames <= frames)
        {
            writeInFileDistances(nFrames, markerIds, markerDistances, realDistance, realAngle, realStepAngle);

            QString sFrames = QString::number(nFrames, 'f', 0);
            ui->framesToDisplay->setText(sFrames);

            if(nFrames == framesToRecord)
            {
                realAngle += realStepAngle;

                QString sDistance = QString::number(realDistance, 'f', 0);
                QString sAngle = QString::number(realAngle, 'f', 0);
                QString sFrames = QString::number(0, 'f', 0);

                ui->framesToDisplay->setText(sFrames);
                ui->distanceToDisplay->setText(sDistance);
                ui->angleToDisplay->setText(sAngle);
            }

        }
        else
        {
            nFrames = 0;
            recordFrames = false;

            for(int k = 0; k<nMarkersDetected.size(); k++)
            {
                nMarkersDetected[k] = 0;
                distAvg[k] = 0;
                errorAvg[k] = 0;
                errorMax[k] = 0;
                errorAvgPerc[k] = 0;
                errorMaxPerc[k] = 0;
            }
        }

        update_window(frame);
        if(waitKey(30) >= 0) break;

        //sleep(1);
    }



    return 1;
}

void MainWindow::writeInFileDistances(int nFrames, vector<int> markerIds, vector<float> markerDistances, int realDistance, int realAngle, int realStepAngle)
{

    QFile file("/home/joao/aruco_tracking_2/markers_distance.txt");

    if(!file.exists())
    {
        qDebug() << file.fileName() << "File does not exist";
    }

    if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append))
    {
        QTextStream txtStream(&file);

        QVector<float> aux(distancesToWrite(nFrames, markerIds, markerDistances, realDistance));

        txtStream << nFrames;
        txtStream << " ";
        txtStream << realDistance;
        txtStream << " ";
        txtStream << realAngle;
        txtStream << " ";
        for (QVector<float>::iterator iter = aux.begin(); iter != aux.end(); iter++){
             txtStream << *iter << " ";
        }
        txtStream << endl;


        file.close();

    }
}

QVector<float> MainWindow::distancesToWrite(int nFrames, vector<int> markerIds, vector<float> markerDistances, int realDistance)
{
    QFile file("/home/joao/aruco_tracking_2/markers_detected.txt");

    QVector<float> ret;

    //markers id used: 2, 3, 4, 5

    for(int id = 2; id < 6; id++)
    {
        IntIterator iter = std::find(markerIds.begin(), markerIds.end(), id);
        if(iter != markerIds.end())
        {
            int index = std::distance(markerIds.begin(), iter);
            ret.push_back(markerDistances[index]*1000);

            //Quantidade de marcadores detetados
            nMarkersDetected[id-2] = nMarkersDetected.at(id-2) + 1;

            //Media das distancias calculadas
            distAvg[id-2] = (distAvg.at(id-2)*(nMarkersDetected.at(id-2)-1) + markerDistances[index]*1000)/nMarkersDetected.at(id-2);

            //Erro medio
            errorAvg[id-2] = (errorAvg.at(id-2)*(nMarkersDetected.at(id-2)-1) + abs(realDistance - markerDistances[index]*1000))/nMarkersDetected.at(id-2);

            //Erro maximo
            if(abs(realDistance - markerDistances[index]*1000) > errorMax.at(id-2))
                errorMax[id-2] = abs(realDistance - markerDistances[index]*1000);
        }
        else
        {
            ret.push_back(0);
        }
    }
    if(nFrames == framesToRecord)
    {
        if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append))
        {
            QTextStream txtStream(&file);
            txtStream << realDistance;
            txtStream << " ";
            txtStream << realAngle;
            txtStream << " ";
            for (QVector<float>::iterator iter = distAvg.begin(); iter != distAvg.end(); iter++){
                 txtStream << *iter << " ";
            }
            for (QVector<float>::iterator iter = nMarkersDetected.begin(); iter != nMarkersDetected.end(); iter++){
                txtStream << *iter << " ";
            }
            for (QVector<float>::iterator iter = errorAvg.begin(); iter != errorAvg.end(); iter++){
                 txtStream << *iter << " ";
            }
            for (QVector<float>::iterator iter = errorMax.begin(); iter != errorMax.end(); iter++){
                 txtStream << *iter << " ";
            }
            txtStream << endl;


            file.close();

        }
        else
        {
            cout << "Can't open the file" << endl;
        }
    }

    return ret;
}

vector<float> MainWindow::checkDifferentMarkers(vector<int> markerIds)
{


    vector<float> ret;

    for(int c=0; c < markerIds.size(); c++)
    {
        if(markerIds[c] == 0 || markerIds[c] == 1)
            ret.push_back(0.080f);
        else if (markerIds[c] == 2 || markerIds[c] == 3)
            ret.push_back(0.120f);
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
            ret.push_back(0.001f);

    }

    return ret;
}

Mat MainWindow::eulerAnglesToRotationMatrix(Vec3d &theta)
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

Vec3f MainWindow::rotationMatrixToEulerAngles(Mat &R)
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

bool MainWindow::isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());

    return  norm(I, shouldBeIdentity) < 1e-6;
}

void MainWindow::on_extractSamples_clicked()
{
    recordFrames = true;
}

void MainWindow::on_stopSamples_clicked()
{
    recordFrames = false;
}

void MainWindow::on_saveData_clicked()
{
    //frames =  ui->frames;
    //realDistance = ui->distance;
    //realAngle = ui->angle;
    //realStepAngle =ui->stepAngle;

    frames = framesToRecord;
    realDistance = 1000;
    realAngle = -75;
    realStepAngle = 5;

    return;
}

void MainWindow::on_increaseDistance_clicked()
{
    realDistance = realDistance + realStepDistance;
    realAngle = -35;

    QString sDistance = QString::number(realDistance, 'f', 0);
    QString sAngle = QString::number(realAngle, 'f', 0);

    ui->distanceToDisplay->setText(sDistance);
    ui->angleToDisplay->setText(sAngle);
}
