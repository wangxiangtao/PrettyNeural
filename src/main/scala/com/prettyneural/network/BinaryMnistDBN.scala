package com.prettyneural.network


import java.io.File
import java.util.ArrayList
import java.io.DataOutputStream
import java.util.Iterator
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.util.Iterator
import com.prettyneural.trainer.StackedRBMTrainer
import com.prettyneural.core.LayerFactory
import com.prettyneural.core.BinaryLayer
import com.prettyneural.core.Layer
import com.prettyneural.core.Tuple
import com.prettyneural.data.minst.MnistDatasetReader
import com.prettyneural.data.minst.MnistItem

class BinaryMnistDBN() {
    var rbm : StackedRBM  = null
    var trainer: StackedRBMTrainer  = null
    var layerFactory = new LayerFactory();
    var  dr: MnistDatasetReader = null
    def this(labels:File, images:File) {
        this()
        dr = new MnistDatasetReader(labels, images);

        rbm = new StackedRBM();
        trainer = new StackedRBMTrainer(rbm, 0.5f, 0.001f, 0.2f, 0.2f, layerFactory);
    }
    
    def learn( iterations: Int,  addLabels: Boolean, stopAt: Int) {

        for (p <- 0 until iterations) {

            // Get random input
            val inputBatch = new ArrayList[Layer]();
            val labelBatch = if(addLabels) new ArrayList[Layer]() else null;


            for (j <- 0 until 30) {
                val trainItem = dr.getTrainingItem();
                val input = layerFactory.create(trainItem.data.length);

                for (i <- 0 until trainItem.data.length)
                    input.set(i, trainItem.data(i));

                inputBatch.add(new BinaryLayer(input));

                if (addLabels) {
                    var labelInput = new Array[Float](10)
                    labelInput(Integer.valueOf(trainItem.label)) = 1.0f;
                    labelBatch.add(layerFactory.create(labelInput));
                }
            }
             var error = trainer.learn(inputBatch, labelBatch, stopAt);
            if (p % 100 == 0)
                System.err.println("Iteration " + p + ", Error = " + error+", Energy = "+rbm.freeEnergy());
        }
    }
    
    def evaluate( test: MnistItem) : Iterator[Tuple]  = {

        var input = layerFactory.create(test.data.length);

        for (i <- 0 until test.data.length)
            input.set(i, test.data(i));

        input = new BinaryLayer(input);

        var stackNum = rbm.getInnerRBMs().size();

        for (i<- 0 until stackNum) {

            val iRBM = rbm.getInnerRBMs().get(i);

            if (iRBM.biasVisible.size() > input.size()) {
                val newInput = new Layer(iRBM.biasVisible.size());

                System.arraycopy(input.get(), 0, newInput.get(), 0, input.size());
                for (j <- input.size() until newInput.size())
                    newInput.set(j, 0.1f);

                input = newInput;
            }

            if (i == (stackNum - 1)) {
                return iRBM.iterator(input);
            }

            input = iRBM.activateHidden(input, null);
        }

        return null;
    }
}

object BinaryMnistDBN {
  
   def start(labels:File,images:File,saveto:File) {

        val m = new BinaryMnistDBN(labels,images);

        var prevStateLoaded = false;

        if (saveto.exists()){
                val input = new DataInputStream(new BufferedInputStream(new FileInputStream(saveto)));
                m.rbm.load(input, m.layerFactory);
                prevStateLoaded = true;
        }


        if (!prevStateLoaded) {
            val numIterations = 1000;

            m.rbm.setLayerFactory(m.layerFactory).addLayer(m.dr.rows * m.dr.cols, false).addLayer(500, false).addLayer(500, false).addLayer(2000, false).withCustomInput(510).build();

            System.err.println("Training level 1");
            m.learn(numIterations, false, 1);
            System.err.println("Training level 2");
            m.learn(numIterations, false, 2);
            System.err.println("Training level 3");
            m.learn(numIterations, true, 3);

            System.out.println("save");
            val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(saveto)));
            m.rbm.save(out);

            out.flush();
            out.close();
        }

        var numCorrect :Double = 0;
        var numWrong  :Double = 0;
        var numAlmost :Double = 0.0;

        while (true) {
            val testCase = m.dr.getTestItem();

            val it = m.evaluate(testCase);

            val labeld = new Array[Float](10);

            for (i <- 0 until 2) {
                val t = it.next();
                var j = (t.visible.size() - 10)
                var k = 0
                while((j < t.visible.size()) && k < 10){
                  labeld(k) += t.visible.get(j);
                  j += 1
                  k += 1
                }
            }

            var max1 = 0.0f;
            var max2 = 0.0f;
            var p1 = -1;
            var p2 = -1;

            System.err.print("Label is: " + testCase.label);


            for (i <- 0 until labeld.length) {
                labeld(i) /= 2;
                if (labeld(i) > max1) {
                    max2 = max1;
                    max1 = labeld(i);
                    p2 = p1;
                    p1 = i;
                }
            }

            System.err.print(", Winner is " + p1 + "(" + max1 + ") second is " + p2 + "(" + max2 + ")");
            if (p1 == Integer.valueOf(testCase.label)) {
                System.err.println(" CORRECT!");
                numCorrect += 1

            } else if (p2 == Integer.valueOf(testCase.label)) {
                System.err.println(" Almost!");
                numAlmost += 1
            } else {
                System.err.println(" wrong :(");
                numWrong += 1
            }
            System.err.println("Test Number:"+(numAlmost + numCorrect + numWrong))
            System.err.println("Error Rate = " + ((numWrong / (numAlmost + numCorrect + numWrong)) * 100));
            System.err.println("Exact Error Rate = " + (numWrong+numAlmost) / (numAlmost + numCorrect + numWrong) * 100);
        }
    }
}