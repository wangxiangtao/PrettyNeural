package com.prettyneural.network

import com.prettyneural.core.LayerFactory
import java.util.ArrayList
import java.awt.Canvas
import com.prettyneural.trainer.SimpleRBMTrainer
import java.util.List
import com.prettyneural.core.BinaryLayer
import java.io.File
import com.prettyneural.core.Layer
import scala.collection.JavaConversions._
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.awt.Graphics2D
import java.awt.Graphics
import com.prettyneural.core.Tuple
import javax.swing.JFrame
import com.prettyneural.data.minst.MnistDatasetReader
import com.prettyneural.data.minst.MnistItem


class BinaryMnistRBM extends Canvas{
  
    var dr:MnistDatasetReader = null
    var rbm:SimpleRBM = null
    val layerFactory = new LayerFactory();
    var trainItem: MnistItem  = null;
    val outputs: List[Array[Int]] = new ArrayList[Array[Int]]
    var trainer: SimpleRBMTrainer = null

    def this(labels: File, images: File) {
        this()
        dr = new MnistDatasetReader(labels, images);
        rbm = new SimpleRBM(dr.cols * dr.rows, 10 * 10, false, layerFactory);
        trainer = new SimpleRBMTrainer(0.2f, 0.001f, 0.2f, 0.1f, layerFactory);
    }

    def learn(): Array[Float] = {
        // Get random input
        val inputBatch: List[Layer] = new ArrayList[Layer]();

        for (j <- 0 until 30) {
            trainItem = dr.getTrainingItem();
            val input = layerFactory.create(trainItem.data.length);

            for (i <- 0 until trainItem.data.length)
                input.set(i, trainItem.data(i));

            inputBatch.add(new BinaryLayer(input));
        }

        val error = trainer.learn(rbm, inputBatch, false); //up down

        if (BinaryMnistRBM.count % 100 == 0)
            System.err.println("Error = " + error + ", Energy = " + rbm.freeEnergy());

        return inputBatch.get(inputBatch.size() - 1).get();
    }
    
    def  evaluate(): Iterator[Tuple] = {

        val test = dr.getTestItem();

        val input = layerFactory.create(test.data.length);

        for (i <- 0 until trainItem.data.length)
            input.set(i, trainItem.data(i));

        return rbm.iterator(new BinaryLayer(input));
    }

    def update() = synchronized {
        learn();
        val it = evaluate();

        outputs.clear();
        for (j <- 0 until 10) {
            val t = it.next();
            val output = new Array[Int](t.visible.size());
            val visible = BinaryLayer.fromBinary(t.visible);

            for (i <- 0 until visible.length) {
                output(i) = Math.round(visible(i));
            }

            outputs.add(output);
        }
        repaint();
    }
    
    override def paint(g:Graphics) = synchronized {

        val in = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);

        if (trainItem != null) {
          var r = in.getRaster();
          r.setDataElements(0, 0, dr.cols, dr.rows, trainItem.data);
          g.drawImage(in, BinaryMnistRBM.border, BinaryMnistRBM.border, null);
  
          var offset = BinaryMnistRBM.border;
          outputs.foreach ( output => {
         
              val out = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);
  
  
              r = out.getRaster();
              r.setDataElements(0, 0, dr.cols, dr.rows, output);
  
              //Resize
              val newImage = new BufferedImage(56, 56, BufferedImage.TYPE_INT_RGB);
  
              val g2 = newImage.createGraphics();
                  g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                          RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                  g2.clearRect(0, 0, 56, 56);
                  g2.drawImage(out, 0, 0, 56, 56, null);
                  g2.dispose();
              g.drawImage(newImage, BinaryMnistRBM.border * 2 + 28, offset, null);
  
              offset += BinaryMnistRBM.border + dr.rows * 2;
          })
  
          var buf = 28 + BinaryMnistRBM.border + BinaryMnistRBM.border;
          for (i <- 0 until rbm.weights.length) {
              if (i % 10 == 0) {
                  offset = BinaryMnistRBM.border;
                  buf += BinaryMnistRBM.border + 56;
              }
  
              var start = new Array[Int](dr.cols * dr.rows)
              for (j <- 0 until start.length)
                  start(j) = if(rbm.weights(i).get(j) > 0) (Math.round(rbm.weights(i).get(j) * 255)) << 8 else ((Math.round(Math.abs(rbm.weights(i).get(j)) * 255)) << 16);
  
              val out = new BufferedImage(dr.cols, dr.rows, BufferedImage.TYPE_INT_RGB);
  
              r = out.getRaster();
              r.setDataElements(0, 0, dr.cols, dr.rows, start);
  
              //Resize
              val newImage = new BufferedImage(56, 56, BufferedImage.TYPE_INT_RGB);
  
              val g2 = newImage.createGraphics();
                  g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                          RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                  g2.clearRect(0, 0, 56, 56);
                  g2.drawImage(out, 0, 0, 56, 56, null);
                  g2.dispose();
              g.drawImage(newImage, buf, offset, null);
  
              offset += BinaryMnistRBM.border + dr.rows * 2;
          }
        }

    }
    
}

object BinaryMnistRBM{
    var border = 10
    var count = 0
    def start( labels: File, images:File) {
        val frame = new JFrame("Mnist Draw");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        val cnvs = new BinaryMnistRBM(labels, images);


        cnvs.setSize(1024, 768);
        frame.add(cnvs);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);


        while (true) {
            cnvs.update();
                count += 1
                if (count > 1000)
                    Thread.sleep(2000);
        }
    }
}
