/**
 *  radius search sample
 * */
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>
#include <random>
#include <chrono>
#include <math.h>

using namespace flann;
using namespace std;

void ShowMatrix(const Matrix<int> &mat){
  for(int i=0;i<mat.rows;i++){
    for(int j=0;j<mat.cols;j++){
      cout<<mat[i][j]<<",";
    }
    cout<<endl;
  }
}


void ShowMatrix(const Matrix<float> &mat){
  for(int i=0;i<mat.rows;i++){
    for(int j=0;j<mat.cols;j++){
      cout<<mat[i][j]<<",";
    }
    cout<<endl;
  }
}

int BruteForceSearch(
    const Matrix<float> &dataset,
    const Matrix<float> &query
    ){
  int ind=-1;
  double dist=10000.0;

  for(int i=0;i<dataset.rows;i++){
      // cout<<mat[i][j]<<",";
      double dx=query[0][0]-dataset[i][0];
      double dy=query[0][1]-dataset[i][1];
      double dz=query[0][2]-dataset[i][2];

      double tdist=sqrt(dx*dx+dy*dy+dz*dz);

      if(tdist<dist){
        dist=tdist;
        ind=i;
      }
  }

  cout<<"Brute dist:"<<dist<<endl;
  return ind;
}

int main(int argc, char** argv)
{
    cout<<"start"<<endl;
    int nDim = 3;
    int nData = 10000;

    Matrix<float> dataset(new float[nDim*nData], nData, nDim);

    random_device rd;
    mt19937 mt(rd());
    
    //datasetを作成
    uniform_real_distribution<float> score(-10.0,10.0);
    for(int i=0;i<nData;i++){
      for(int j=0;j<nDim;j++){
        dataset[i][j]=score(mt);
        // cout<<dataset[i][j]<<endl;
      }
    }

    Matrix<float> query(new float[nDim*1],1, nDim);
    query[0][0]=10.0;
    query[0][1]=2.0;
    query[0][2]=10.0;
    // cout<<"query:";
    // ShowMatrix(query);

    Matrix<int> indices(new int[query.rows],query.rows,1);
    Matrix<float> dists(new float[query.rows],query.rows, 1);

    // cout<<query.rows<<endl;
    Index<L2<float> > index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();                                                                                               
    // index.knnSearch(query, indices, dists, nDim, flann::SearchParams(128));
    chrono::system_clock::time_point  start, end; // 型は auto で可

    //Bruteforce
    start = chrono::system_clock::now(); // 計測開始時間)
    int ind=BruteForceSearch(dataset,query);
    cout<<"brute:ind:"<<ind<<endl;
    end = chrono::system_clock::now();  // 計測終了時間
    double msec = (double)chrono::duration_cast<chrono::nanoseconds>(end-start).count()/1000.0;
    cout<<"time:"<<msec<<endl;

    start = chrono::system_clock::now(); // 計測開始時間)
    cout<<index.radiusSearch(query, indices, dists, 100.0,flann::SearchParams(128))<<endl;
    end = chrono::system_clock::now();  // 計測終了時間
    msec = (double)chrono::duration_cast<chrono::nanoseconds>(end-start).count()/1000.0;
    cout<<"time:"<<msec<<endl;

    cout<<"indices:";
    ShowMatrix(indices);
    cout<<"dists:";
    ShowMatrix(dists);
    for(int i=0;i<indices.cols;i++){
      // cout<<indices[0][i]<<endl;
      cout<<dataset[indices[0][i]][0]<<",";
      cout<<dataset[indices[0][i]][1]<<",";
      cout<<dataset[indices[0][i]][2]<<endl;
    }

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    
    return 0;
}
