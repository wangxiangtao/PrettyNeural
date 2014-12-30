package com.prettyneural.dbn

import java.io.File
import com.prettyneural.network.BinaryMnistDBN
import com.prettyneural.network.BinaryMnistRBM
import org.junit.Test

class TestNetwork {
  @Test
  def testDBN(): Unit = {
        var labels = new File("target/mnist/train-labels-idx1-ubyte.gz")
        var images = new File("target/mnist/train-images-idx3-ubyte.gz")
        var saveto = new File("dbn.bin")
        
        BinaryMnistDBN.start(labels, images, saveto)
  }    
  
  @Test
  def testRBM() {
        var labels = new File("target/mnist/train-labels-idx1-ubyte.gz")
        var images = new File("target/mnist/train-images-idx3-ubyte.gz")
        
        BinaryMnistRBM.start(labels,images)
  }
}