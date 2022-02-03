import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' show Color;
import 'package:flutter/services.dart';

class Tflite {
  static const MethodChannel _channel = const MethodChannel('tflite');

  static Future<String?> loadModel(
      {required String model,
      String labels = "",
      int numThreads = 1,
      bool isAsset = true,
      bool useGpuDelegate = false}) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": model,
        "labels": labels,
        "numThreads": numThreads,
        "isAsset": isAsset,
        'useGpuDelegate': useGpuDelegate
      },
    );
  }

  static Future<List?> runModelOnImage(
      {required String path,
      bool asynch = true}) async {
    return await _channel.invokeMethod(
      'runModelOnImage',
      {
        "path": path,
        "asynch": asynch,
      },
    );
  }

  static Future<List?> runModelOnFrame(
      {required ByteBuffer byteBuffer,
      bool asynch = true}) async {
    return await _channel.invokeMethod(
      'runModelOnFrame',
      {
        "byteBuffer": byteBuffer,
        "asynch": asynch,
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }
}
