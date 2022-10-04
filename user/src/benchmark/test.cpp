//
// Created by root on 22-9-19.
//
#include<pybind11/pybind11.h>
#include<cstdlib>
#include<cstdio>
#include<ctime>
#include<string>
#include "../install_linux-x86_64_avx512_vnni/include/c/bolt.h"
#include<pybind11/numpy.h>
#include<fstream>
#include <pybind11/eigen.h>
#include<iostream>
#include <Eigen/Dense>
namespace py = pybind11;
using namespace std;
using namespace Eigen;
using RowMatrixXd = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// Use RowMatrixXd instead of MatrixXd
class Dog{
private:
    int counter;

    AFFINITY_TYPE affinity = CPU; // CPU_HIGH_PERFORMANCE

    ModelHandle model;
    ResultHandle result;
    float **data;
    int num_inputs;
    char **name;
    float** outputData;

public:
    Dog(const char*s){counter=0;};
    Dog(){counter=0;};
    int prepare(char*path){
//        char*s = "/home/data/cypo/bolt/test.bolt";
        model = CreateModel(path,affinity,NULL);
        num_inputs = GetNumInputsFromModel(model);
        if(num_inputs==0)return 0;
        name = (char**) malloc(sizeof(char*));
        *name = (char*)malloc(sizeof(char)*128);
        int n,c,h,w;
        DATA_TYPE dt;
        DATA_FORMAT df;
        double total_time, inf_time;
        GetInputDataInfoFromModel(model,num_inputs,name,&n,&c,&h,&w,&dt,&df);

        PrepareModel(model,num_inputs,(const char**)name,&n,&c,&h,&w,&dt,&df);
        result = AllocAllResultHandle(model);

        outputData = (float **)malloc(sizeof(float *) * 2);
        return 0;
    }
    float inference(Eigen::Ref<const RowMatrixXd>& A_mat, Eigen::Ref<Eigen::VectorXd> v){
        const float* arr123 = A_mat.data();
        auto **t = reinterpret_cast<const float **>(malloc(sizeof(void *)));
        *t = arr123;
        clock_t start_0 = clock();
        RunModel(model, result, num_inputs, (const char **) name, (void **) t);
        clock_t rmt = clock() - start_0;
//        printf("c infer time ============= %f\n", ((float)rmt / CLOCKS_PER_SEC));
//        printf("rmt: %.2lf ms\n", rmt / 1000.0);
        GetOutputDataFromResultHandle(result, 2, (void**)outputData);
        v[0] = outputData[0][0];
        for(int j = 0; j < 4; j++){
            v[j+1] = outputData[1][j];
        }
        return (float)((float)rmt/1000.0);
    }
    int run(){
        return counter++;
    }
    static Dog& instance(){
        static const char* default_model_path = "./test.bolt";
        static Dog dog(default_model_path);
        return dog;
    }
};


PYBIND11_PLUGIN(example) {
    py::module m("example", "pybind11 example plugin");
// Use this function to get access to the singleton
    m.def("get_instance",
          &Dog::instance,
          py::return_value_policy::reference,
          "Get reference to the singleton");

    py::class_<Dog>(m, "Dog")
            // No init!
            .def(py::init<>())
            .def("run",
                 &Dog::run,
                 "test"
                )
            .def("prepare",
                 &Dog::prepare,
                 "test"
                 )
            .def("inference",
                  &Dog::inference,
                  "test"
              )
            .def("get_instance",
                   &Dog::instance,
                   "test")
            ;

    py::object gg = py::cast(new Dog());
    m.attr("s1") = gg;
//    py::object gm = py::cast(Dog[]);
    return m.ptr();
}