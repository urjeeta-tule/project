package com.example.anuj.calculator;

import android.util.Log;

import java.lang.Math.*;

/**
 * Created by anuj on 14/4/17.
 */

class Calculator {

    public Double saveMem;
    public Calculator(){
        double saveMem = 0;
    }

    double Add(double num1, double num2){
        return num1+num2;
    }

    double Sub(double num1, double num2){
        return num1-num2;
    }

    double Mul(double num1, double num2){
        return num1*num2;
    }

    double Div(double num1, double num2){
        return num1/num2;
    }

    double Sin(double num1){
        return Math.asin(Math.toRadians(num1));
    }

    double Cos(double num1){
        return Math.acos(Math.toRadians(num1));
    }

    double Tan(double num1){
        return Math.atan(Math.toRadians(num1));
    }

    double SqRt(double num1){
        return Math.sqrt(num1);
    }

    void Save(double num1){
        saveMem = num1;
    }

    Double Recall(){
        return saveMem;
    }

    void ClearMem(){
        try{
            saveMem = null;
        } catch (Exception NullPointerException){
            //Number empty
        }
    }
}
