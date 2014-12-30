package com.prettyneural.core

import java.util.Arrays

class Layer {
  var layer: Array[Float] = null
  
  def this(layer: Array[Float]) {
    this()
    this.layer = layer
  }
  
  def this(size: Int) {
    this()
    layer = new Array[Float](size)
  } 
  
  def set(i: Int, f: Float) = { layer(i) = f }

  def get(i: Int) = layer(i)

  def add(i: Int, f: Float) = layer(i) += f

  def div(i: Int, f: Float) = layer(i) /= f

  def mult(i: Int, f: Float) = layer(i) *= f

  def cloneLayer = {
    var c = new Layer(layer.length)
    System.arraycopy(layer, 0, c.layer, 0, layer.length)
    c
  }

  def clear = Arrays.fill(layer, 0.0f)

  def copy(src: Array[Float]) {
    System.arraycopy(layer, 0, src, 0, layer.length)
  }

  def get() = layer

  def size() = layer.length
}