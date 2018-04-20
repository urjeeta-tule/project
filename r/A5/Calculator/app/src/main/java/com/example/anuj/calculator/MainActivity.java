package com.example.anuj.calculator;

import android.content.Context;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

public class MainActivity extends AppCompatActivity implements android.view.View.OnClickListener {

    Calculator calc = new Calculator();
    EditText num1EditText;
    EditText num2EditText;
    TextView answerTextView;
    Double num1;
    Double num2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //edittexts
        num1EditText = (EditText) findViewById(R.id.editText1);
        num2EditText = (EditText) findViewById(R.id.editText2);

        //textviews
        answerTextView = (TextView) findViewById(R.id.answertextView);

        //button listeners
        findViewById(R.id.buttonAdd).setOnClickListener(this);
        findViewById(R.id.buttonSub).setOnClickListener(this);
        findViewById(R.id.buttonDiv).setOnClickListener(this);
        findViewById(R.id.buttonMul).setOnClickListener(this);
        findViewById(R.id.buttonSin).setOnClickListener(this);
        findViewById(R.id.buttonCos).setOnClickListener(this);
        findViewById(R.id.buttonTan).setOnClickListener(this);
        findViewById(R.id.buttonSqRt).setOnClickListener(this);
        findViewById(R.id.buttonSave).setOnClickListener(this);
        findViewById(R.id.buttonRecall).setOnClickListener(this);
        findViewById(R.id.buttonClear).setOnClickListener(this);
        findViewById(R.id.buttonFileRead).setOnClickListener(this);
        findViewById(R.id.buttonFileWrite).setOnClickListener(this);


    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {

            case R.id.buttonAdd:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    num2 = Double.parseDouble(num2EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Add(num1, num2)));
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter numbers in input fields", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonSub:
                try {
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    num2 = Double.parseDouble(num2EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Sub(num1, num2)));
                }catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter numbers in input fields", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonDiv:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    num2 = Double.parseDouble(num2EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Div(num1, num2)));
                }catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter numbers in input fields", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonMul:
                try {
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    num2 = Double.parseDouble(num2EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Mul(num1, num2)));
                }catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter numbers in input fields", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonSin:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Sin(num1)));
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter number in first input field", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonCos:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Cos(num1)));
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter number in first input field", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonTan:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.Tan(num1)));
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter number in first input field", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonSqRt:
                try{
                    num1 = Double.parseDouble(num1EditText.getText().toString());
                    answerTextView.setText(String.valueOf(calc.SqRt(num1)));
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter number in first input field", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonSave:
                try{
                    calc.Save(Double.parseDouble(num1EditText.getText().toString()));
                    Toast.makeText(getApplicationContext(), "Number Saved in memory", Toast.LENGTH_SHORT).show();
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "Enter number in first input field", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.buttonRecall:
                try{
                    double numToSave = calc.Recall();
                    num1EditText.setText(Double.toString(numToSave));
                    Toast.makeText(getApplicationContext(), "Number extracted from memory", Toast.LENGTH_SHORT).show();
                } catch (Exception NumberFormatException){
                    Toast.makeText(getApplicationContext(), "No number saved in memory", Toast.LENGTH_SHORT).show();
                }
                break;
            case R.id.buttonClear:
                Double numToSave = calc.Recall();
                if(numToSave==null){
                    Toast.makeText(getApplicationContext(), "No number saved in memory", Toast.LENGTH_SHORT).show();
                }else{
                    calc.ClearMem();
                    Toast.makeText(getApplicationContext(), "Number Cleared", Toast.LENGTH_SHORT).show();
                }
                break;

            //this part is for A5
            case R.id.buttonFileRead:
                //do file read and show stuff on screen
                try{
                    File file = new File(Environment.getExternalStorageDirectory(), "Numbers.xml");
                    FileWriter fileWriter = new FileWriter(file);
                    BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
                    String writetofile = "<numbers><num>15</num><num>5</num></numbers>";
                    bufferedWriter.write(writetofile);
                    bufferedWriter.close();

                    //this commented part to read from normal file, original content to read from xml file
                    // BufferedReader br = new BufferedReader(new FileReader(file));
                    // String st;
                    // while((st=br.readLine()) != null){
                    //     System.out.println(st);
                    // }

                    DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
                    DocumentBuilder dbuilder = documentBuilderFactory.newDocumentBuilder();
                    Document doc = dbuilder.parse(file);
                    NodeList nodeList = doc.getElementsByTagName("num");
                    Double num1 = Double.parseDouble(nodeList.item(0).getTextContent());
                    Double num2 = Double.parseDouble(nodeList.item(1).getTextContent());;
                    double answer = calc.Add(num1, num2);
                    num1EditText.setText(String.valueOf(num1));
                    num2EditText.setText(String.valueOf(num2));
                    answerTextView.setText(String.valueOf(answer));

                    Toast.makeText(getApplicationContext(), "Values read from file", Toast.LENGTH_SHORT).show();
                }catch (Exception e){
                    Toast.makeText(getApplicationContext(), "did you give storage permission?", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                }



                break;

            case R.id.buttonFileWrite:
                //save answer to file
                try {
                    String answer = (String) answerTextView.getText();
                    File file = new File(Environment.getExternalStorageDirectory(),"Answer.xml");
                    FileWriter fileWriter = new FileWriter(file);
                    BufferedWriter bwritter = new BufferedWriter(fileWriter);
                    String writetofile = "<answer>".concat(answer).concat("</answer>");
                    bwritter.write(writetofile);
                    bwritter.close();
                    Toast.makeText(getApplicationContext(), "File Written", Toast.LENGTH_SHORT).show();
                } catch (Exception e){
                    e.printStackTrace();
                    Toast.makeText(getApplicationContext(), "Did you give Storage Permissions?", Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                Toast.makeText(getApplicationContext(), "This button got no Listener", Toast.LENGTH_SHORT).show();
                break;
        }
    }
}
