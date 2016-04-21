#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "opencv2/core/ocl.hpp"
 

using namespace cv;
using namespace std;


int main (int argc,char **argv)
{
    // Refernce http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf (IEEE PAMI v12 n 7 1990)
//Mat x = imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/1441883251877551.png",CV_LOAD_IMAGE_GRAYSCALE);
Mat x = imread("f:/lib/opencv/samples/data/lena.jpg",CV_LOAD_IMAGE_GRAYSCALE);
Mat x0;
x.convertTo(x0, CV_32FC1);
 

double t=0;
double lambda=0.25; // Defined in equation (7)
double K=10,K2=(1/K/K); // defined after equation(13) in text
imshow("Original",x);
Mat    dI00 = Mat::zeros(x0.size(),CV_32F);

Mat x1,xc;

while (t<1000)
{
    Mat D; // defined just before equation (5) in text
    Mat gradxX,gradyX; // Image Gradient t time 
    Sobel(x0,gradxX,CV_32F,1,0,3);
    Sobel(x0,gradyX,CV_32F,0,1,3);
    D = Mat::zeros(x0.size(),CV_32F);
    for (int i=0;i<x0.rows;i++)
        for (int j = 0; j < x0.cols; j++)
        {
            float gx = gradxX.at<float>(i, j), gy = gradyX.at<float>(i,j);
            float d;
            if (i==0 || i== x0.rows-1 || j==0 || j==x0.cols-1) // conduction coefficient set to 1 p633 after equation 13
                d=1;
            else
                d =1.0/(1+(gx*gx+0*gy*gy)*K2); // expression of g(gradient(I))
                //d =-exp(-(gx*gx+gy*gy)*K2); // expression of g(gradient(I))
            D.at<float>(i, j) = d;
       }
    x1 = Mat::zeros(x0.size(),CV_32F);
    double maxD=0,intxx=0;
    {
        int i=0;
        float *u1 = (float*)x1.ptr(i);
        u1++;
        for (int j = 1; j < x0.cols-1; j++,u1++)
            {
                // Value of I at (i+1,j),(i,j+1)...(i,j)
                float ip10=x0.at<float>(i+1, j),i0p1=x0.at<float>(i, j+1);
                float i0m1=x0.at<float>(i, j-1),i00=x0.at<float>(i, j);
                // Value of D at at (i+1,j),(i,j+1)...(i,j)
                float cp10=D.at<float>(i+1, j),c0p1=D.at<float>(i, j+1);
                float c0m1=D.at<float>(i, j-1),c00=D.at<float>(i, j);
                // Equation (7) p632
                double xx=(cp10+c00)*(ip10-i00) + (c0p1+c00)*(i0p1-i00) + (c0m1+c00)*(i0m1-i00);
                dI00.at<float>(i, j) = xx;
                if (maxD<fabs(xx))
                    maxD=fabs(xx);
                intxx+=fabs(xx);
                // equation (9)
           }
    }
    for (int i = 1; i < x0.rows-1; i++)
    {
        float *u1 = (float*)x1.ptr(i);
        int j=0;
        if (j==0)
        {
            // Value of I at (i+1,j),(i,j+1)...(i,j)
            float ip10=x0.at<float>(i+1, j),i0p1=x0.at<float>(i, j+1);
            float im10=x0.at<float>(i-1, j),i00=x0.at<float>(i, j);
            // Value of D at at (i+1,j),(i,j+1)...(i,j)
            float cp10=D.at<float>(i+1, j),c0p1=D.at<float>(i, j+1);
            float cm10=D.at<float>(i-1, j),c00=D.at<float>(i, j);
            // Equation (7) p632
            double xx=(cp10+c00)*(ip10-i00) + (c0p1+c00)*(i0p1-i00) + (cm10+c00)*(im10-i00);
            dI00.at<float>(i, j) = xx;
            if (maxD<fabs(xx))
                maxD=fabs(xx);
            intxx+=fabs(xx);
            // equation (9)
       }

        u1++;
        j++;
        for (int j = 1; j < x0.cols-1; j++,u1++)
        {
            // Value of I at (i+1,j),(i,j+1)...(i,j)
            float ip10=x0.at<float>(i+1, j),i0p1=x0.at<float>(i, j+1);
            float im10=x0.at<float>(i-1, j),i0m1=x0.at<float>(i, j-1),i00=x0.at<float>(i, j);
            // Value of D at at (i+1,j),(i,j+1)...(i,j)
            float cp10=D.at<float>(i+1, j),c0p1=D.at<float>(i, j+1);
            float cm10=D.at<float>(i-1, j),c0m1=D.at<float>(i, j-1),c00=D.at<float>(i, j);
            // Equation (7) p632
            double xx=(cp10+c00)*(ip10-i00) + (c0p1+c00)*(i0p1-i00) + (cm10+c00)*(im10-i00)+ (c0m1+c00)*(i0m1-i00);
            dI00.at<float>(i, j) = xx;
            if (maxD<fabs(xx))
                maxD=fabs(xx);
            intxx+=fabs(xx);
            // equation (9)
        }
        j++;
        if (j==x0.cols-1)
        {
            // Value of I at (i+1,j),(i,j+1)...(i,j)
            float ip10=x0.at<float>(i+1, j);
            float im10=x0.at<float>(i-1, j),i0m1=x0.at<float>(i, j-1),i00=x0.at<float>(i, j);
            // Value of D at at (i+1,j),(i,j+1)...(i,j)
            float cp10=D.at<float>(i+1, j);
            float cm10=D.at<float>(i-1, j),c0m1=D.at<float>(i, j-1),c00=D.at<float>(i, j);
            // Equation (7) p632
            double xx=(cp10+c00)*(ip10-i00)  + (cm10+c00)*(im10-i00)+ (c0m1+c00)*(i0m1-i00);
            dI00.at<float>(i, j) = xx;
            if (maxD<fabs(xx))
                maxD=fabs(xx);
            intxx+=fabs(xx);
            // equation (9)
       }
    }
    {
        int i=x0.rows-1;
        float *u1 = (float*)x1.ptr(i);
        u1++;
        for (int j = 1; j < x0.cols-1; j++,u1++)
        {
            // Value of I at (i+1,j),(i,j+1)...(i,j)
            float i0p1=x0.at<float>(i, j+1);
            float im10=x0.at<float>(i-1, j),i0m1=x0.at<float>(i, j-1),i00=x0.at<float>(i, j);
            // Value of D at at (i+1,j),(i,j+1)...(i,j)
            float c0p1=D.at<float>(i, j+1);
            float cm10=D.at<float>(i-1, j),c0m1=D.at<float>(i, j-1),c00=D.at<float>(i, j);
            // Equation (7) p632
            double xx= (c0p1+c00)*(i0p1-i00) + (cm10+c00)*(im10-i00)+ (c0m1+c00)*(i0m1-i00);
            dI00.at<float>(i, j) = xx;
            if (maxD<fabs(xx))
                maxD=fabs(xx);
            intxx+=fabs(xx);
            // equation (9)
       }
    }
    lambda=100/maxD;
    cout <<" lambda = "<< lambda<<"\t Maxd"<<maxD << "\t"<<intxx<<"\n";
    for (int i = 0; i < x0.rows; i++)
    {
        float *u1 = (float*)x1.ptr(i);
        for (int j = 0; j < x0.cols; j++,u1++)
        {
            *u1 = x0.at<float>(i, j) + lambda/4*dI00.at<float>(i, j);
            // equation (9)
       }
    }
    x1.copyTo(x0);
    x0.convertTo(xc,CV_8U);
    imshow("Perrony x0",xc);
    cout << "*";
    char c=waitKey(10);
    if (c==27)
        break;
    t=t+lambda;
}

    imwrite("perrona.png",xc);


return 0;
 }