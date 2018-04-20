package com.example.anuj.calculator;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {

    public Calculator calc = new Calculator();
    double value1 = 15.5;
    double value2 = 5.0;
    double value3 = 45;
    double answer;
    @Test
    public void addition_isCorrect() {
        answer = calc.Add(value1, value2);
        assertEquals(20.5, answer, 0.001);
    }

    @Test
    public void subtraction_isCorrect() throws Exception {
        answer = calc.Sub(value1, value2);
        assertEquals(10.5, answer, 0.001);
    }

    @Test
    public void multiplication_isCorrect() throws Exception {
        answer = calc.Mul(value1, value2);
        assertEquals(77.5, answer, 0.001);
    }

    @Test
    public void division_isCorrect() throws Exception {
        answer = calc.Div(value1, value2);
        assertEquals(3.1, answer, 0.001);
    }

    @Test
    public void sine_isCorrect() throws Exception {
        answer = calc.Sin(value3);
        assertEquals(0.9033, answer, 0.001);
    }

    @Test
    public void cosine_isCorrect() throws Exception {
        answer = calc.Cos(value3);
        assertEquals(0.6674, answer, 0.001);
    }

    @Test
    public void tangent_isCorrect() throws Exception {
        answer = calc.Tan(value3);
        assertEquals(0.6667, answer, 0.001);
    }

    @Test
    public void SquareRoot_isCorrect() throws Exception {
        answer = calc.SqRt(value3);
        assertEquals(6.7082, answer, 0.001);
    }

    @Test
    public void SaveRecall_isCorrect() throws Exception {
        calc.Save(value3);
        double checkSaveNum = calc.Recall();
        assertEquals(value3, checkSaveNum, 0);
    }

    @Test
    public void Clear_isCorrect() throws Exception {
        calc.ClearMem();
        Double checkSaveNum = calc.Recall();
        assertEquals(null, checkSaveNum);
    }


}