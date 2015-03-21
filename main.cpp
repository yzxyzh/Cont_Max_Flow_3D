#include <iostream>
#include "ContMaxFlow.h"
#include <armadillo>
#include <stdexcept>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>
#include <itkMetaImageIO.h>
#include <itkImageIOBase.h>
#include <limits>

using namespace arma;

void GenerateGaussianKernel(int half_width, double sigma, cube& output)
{
    int width = half_width*2+1;
    output.reshape(width, width, width);
    
    //output.create(width, width, CV_64FC1);
    //output.setTo(Scalar(0));
    for (int u=0;u<=half_width; u++) {
        for (int v=0; v<=half_width; v++) {
            for (int w=0; w<=half_width; w++) {
                
                for (int alpha = -1; alpha<=1; alpha+=2) {
                    for (int beta = -1; beta<=1; beta+=2) {
                        for (int gamma = -1; gamma<=1; gamma+=2) {
                            output(half_width+alpha*u,half_width+beta*v,half_width+gamma*w) = exp(-(u*u+v*v+w*w)/(2.0*sigma*sigma));

                        }
                    }
                }
                
            }

        }
    }
    
    
}

void BinaryGreaterThan(cube& left,double z,cube& output)
{
    output.copy_size(left);
    
    cube::iterator l_start = left.begin();
    cube::iterator l_end = left.end();
    cube::iterator out_start = output.begin();
    
    for (cube::iterator i=l_start; i!=l_end; ++i,++out_start) {
        if(*i>z)
            *out_start = 1.0;
        else
            *out_start = 0.0;
    }
}

void BinarySmallerThan(cube& left,cube& right,cube& output)
{
    if(left.size()!=right.size())
        throw std::runtime_error("my cube min: input size mismatch");
    
    output.copy_size(left);
    
    cube::iterator l_start = left.begin();
    cube::iterator l_end = left.end();
    cube::iterator r = right.begin();
    cube::iterator out = output.begin();
    
    for (cube::iterator i=l_start; i!=l_end; ++i,++r,++out) {
        if(*i>*r)
            *out = 0;
        else
            *out = 1.0;
    }

    
    
}

void BinaryBiggerThan(cube& left,cube& right,cube& output)
{
    if(left.size()!=right.size())
        throw std::runtime_error("my cube min: input size mismatch");
    
    output.copy_size(left);
    
    cube::iterator l_start = left.begin();
    cube::iterator l_end = left.end();
    cube::iterator r = right.begin();
    cube::iterator out = output.begin();
    
    for (cube::iterator i=l_start; i!=l_end; ++i,++r,++out) {
        if(*i<*r)
            *out = 0;
        else
            *out = 1.0;
    }
    
    
    
}

//output = min(l,r);
void myMin(cube& left,cube& right,cube& output)
{
    if(left.size()!=right.size())
        throw std::runtime_error("my cube min: input size mismatch");
    
    output.copy_size(left);
    
    cube::iterator l_start = left.begin();
    cube::iterator l_end = left.end();
    cube::iterator r = right.begin();
    cube::iterator out = output.begin();
    
    for (cube::iterator i=l_start; i!=l_end; ++i,++r,++out) {
        if(*i>*r)
            *out = *r;
        else
            *out = *i;
    }

}

double l1_norm(cube& in)
{
    cube::iterator _start = in.begin();
    cube::iterator _end = in.end();
    
    double norm = 0;
    
    for (cube::iterator i=_start; i!=_end; ++i) {
        norm+=fabs(*i);
    }
    
    return norm;
}

int main()
{
    cube erru;
    const double eps_min = std::numeric_limits<double>::epsilon();
    
    typedef itk::Image<double,3> itk_3d_double;
    typedef itk::ImageFileReader<itk_3d_double> ReaderType;
    itk::NiftiImageIO::Pointer niftIO = itk::NiftiImageIO::New();
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetImageIO(niftIO);
    reader->SetFileName("/Users/yanzixu/Downloads/IM_0190_frame_01.nii");
    reader->Update();
    
    itk_3d_double::Pointer outcome = reader->GetOutput();
    itk_3d_double::SizeType size = outcome->GetLargestPossibleRegion().GetSize();
    itk_3d_double::IndexType coord;
    
    cube ur(size[0],size[1],size[2]);
    
    
    for (int z=0; z<size[2]; z++) {
        coord[2]=z;
        for (int y=0; y<size[1]; y++) {
            coord[1]=y;
            for (int x=0; x<size[0]; x++) {
                coord[0] = x;
                ur(x,y,z) = outcome->GetPixel(coord)/255.0;
            }
        }
    }
    
    cout<<"reading complete"<<endl;
    
    //GenerateGaussianKernel(2, 2.0, ur);
    
    //cout<<ur<<endl;
    //getchar();
    
    int rows = ur.n_rows;
    int cols = ur.n_cols;
    int heights = ur.n_slices;
    
    int szVol = rows*cols*heights;
    
    cout<<"Rows = "<<rows<<" Cols = "<<cols<<" Heights = "<<heights<<endl;
    cout<<" Volume = "<<szVol<<endl;
//    getchar();
    
    
    cube alpha = 0.2*arma::ones(rows, cols, heights);
    double cc = 0.35;
    double errbound = 5e-4;
    int numIter = 300;
    double steps = 0.11;
    
    double ulab[2]={0.2,0.7};
    
    cube Cs = arma::abs(ur-ulab[0]);
    cube Ct = arma::abs(ur-ulab[1]);
    
    cube cs_m_ct = Cs-Ct;
    cube u;
    BinaryGreaterThan(cs_m_ct, 0, u);
    
//    cout<<Cs<<endl;
//    cout<<endl;
//    cout<<Ct<<endl;
//    cout<<endl;
//    cout<<u<<endl;
//    cout<<endl;
//    getchar();
    
    
    cube ps;
    myMin(Ct, Cs, ps);
    
    cube pt = ps;
    
    cube pp1 = arma::zeros(rows,cols+1,heights);
    cube pp2 = arma::zeros(rows+1,cols,heights);
    cube pp3 = arma::zeros(rows,cols,heights+1);
    
    cube divp = pp2.subcube(1, 0, 0, rows, cols-1, heights-1)-pp2.subcube(0, 0, 0, rows-1, cols-1, heights-1)
               +pp3.subcube(0, 0, 1, rows-1, cols-1, heights)-pp3.subcube(0, 0, 0, rows-1, cols-1, heights-1)
               +pp1.subcube(0, 1, 0, rows-1, cols, heights-1)-pp1.subcube(0, 0, 0, rows-1, cols-1, heights-1);
    
//    cout<<ps<<endl;
//    cout<<divp<<endl;
//    getchar();
    
    double erriter = 0.0;
    
    int i = 0;
    for (i=0; i<numIter; i++) {
        
        cout<<"iter "<<i<<"   ";
        
        cube pts = divp - (ps - pt + u/cc);
        pp1.subcube(0, 1, 0, rows-1, cols-1, heights-1)+=
        steps*(pts.subcube(0, 1, 0, rows-1, cols-1, heights-1)-pts.subcube(0, 0, 0, rows-1, cols-2, heights-1));
        
        pp2.subcube(1, 0, 0, rows-1, cols-1, heights-1)+=
        steps*(pts.subcube(1, 0, 0, rows-1, cols-1, heights-1)-pts.subcube(0, 0, 0, rows-2, cols-1, heights-1));
        
        pp3.subcube(0, 0, 1, rows-1, cols-1, heights-1)+=
        steps*(pts.subcube(0, 0, 1, rows-1, cols-1, heights-1)-pts.subcube(0, 0, 0, rows-1, cols-1, heights-2));
        
        
//        cout<<pp1<<endl;
//        cout<<endl;
//        cout<<pp2<<endl;
//        cout<<endl;
//        cout<<pp3<<endl;
//        cout<<endl;
//        getchar();
        //compute gk
        
        cube gk = 0.5*(
              pow(pp1.subcube(0, 0, 0, rows-1, cols-1, heights-1),2.0)+pow(pp1.subcube(0, 1, 0, rows-1, cols, heights-1),2.0)
            + pow(pp2.subcube(0, 0, 0, rows-1, cols-1, heights-1),2.0)+pow(pp2.subcube(1, 0, 0, rows, cols-1, heights-1),2.0)
            + pow(pp3.subcube(0, 0, 0, rows-1, cols-1, heights-1),2.0)+pow(pp3.subcube(0, 0, 1, rows-1, cols-1, heights),2.0)
        
        
        
        );
        
        gk = pow(gk,0.5);
        
        cube gk_l_alpha;
        BinarySmallerThan(gk,alpha,gk_l_alpha);
        cube gk_g_alpha;
        BinaryBiggerThan(gk, alpha, gk_g_alpha);
        gk = gk_l_alpha+gk_g_alpha%gk%(1.0/(alpha+eps_min));
        gk = 1.0/(gk+eps_min);
        
        pp1.subcube(0, 1, 0, rows-1, cols-1, heights-1)%=
        ( 0.5*(gk.subcube(0, 1, 0, rows-1, cols-1, heights-1)+gk.subcube(0, 0, 0, rows-1, cols-2, heights-1)) );

        pp2.subcube(1, 0, 0, rows-1, cols-1, heights-1)%=
        ( 0.5*(gk.subcube(1, 0, 0, rows-1, cols-1, heights-1)+gk.subcube(0, 0, 0, rows-2, cols-1, heights-1)) );
        
        pp3.subcube(0, 0, 1, rows-1, cols-1, heights-1)%=
        ( 0.5*(gk.subcube(0, 0, 1, rows-1, cols-1, heights-1)+gk.subcube(0, 0, 0, rows-1, cols-1, heights-2)) );
        
        
        divp = pp2.subcube(1, 0, 0, rows, cols-1, heights-1)-pp2.subcube(0, 0, 0, rows-1, cols-1, heights-1)
              +pp3.subcube(0, 0, 1, rows-1, cols-1, heights)-pp3.subcube(0, 0, 0, rows-1, cols-1, heights-1)
              +pp1.subcube(0, 1, 0, rows-1, cols, heights-1)-pp1.subcube(0, 0, 0, rows-1, cols-1, heights-1);
        
        
        pts = divp - u/cc + pt + 1.0/cc;
        
        myMin(pts, Cs, ps);
        
        pts = -divp + ps + u/cc;
        
        myMin(pts, Ct, pt);

        erru = cc*(divp + pt - ps);
        
        u = u - erru;
        
        erriter = l1_norm(erru)/szVol;
        
        cout<<"erriter = "<<erriter<<endl;
        
        if(erriter<errbound) break;
        
        
    }
    
    //cout<<erru<<endl;
    
    cout<<i<<endl;
    //cout<<u<<endl;
    
    
    
    
    
    cout<<"finished"<<endl;
    
    
    
    
    
    
    
    
    
    
    
    
    return EXIT_SUCCESS;
}