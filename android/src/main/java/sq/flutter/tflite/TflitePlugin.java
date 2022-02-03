package sq.flutter.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.Tensor;

//import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


public class TflitePlugin implements MethodCallHandler {
  private final Registrar mRegistrar;
  private Interpreter tfLite;
  private boolean tfLiteBusy = false;
  private int inputSize = 0;

  float[] output;
  private static final int BYTES_PER_CHANNEL = 4;

  Map<String, Integer> partsIds = new HashMap<>();
  List<Integer> parentToChildEdges = new ArrayList<>();
  List<Integer> childToParentEdges = new ArrayList<>();

  public static void registerWith(Registrar registrar) {
    final MethodChannel channel = new MethodChannel(registrar.messenger(), "tflite");
    channel.setMethodCallHandler(new TflitePlugin(registrar));
  }

  private TflitePlugin(Registrar registrar) {
    this.mRegistrar = registrar;
  }

  @Override
  public void onMethodCall(MethodCall call, Result result) {
    if (call.method.equals("loadModel")) {
      try {
        String res = loadModel((HashMap) call.arguments);
        result.success(res);
      } catch (Exception e) {
        result.error("Failed to load model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnImage")) {
      try {
        new RunModelOnImage((HashMap) call.arguments, result).executeTfliteTask();
      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnFrame")) {
      try {
        new RunModelOnFrame((HashMap) call.arguments, result).executeTfliteTask();
      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else {
      result.error("Invalid method", call.method.toString(), "");
    }
  }

  private String loadModel(HashMap args) throws IOException {
    String model = args.get("model").toString();
    Object isAssetObj = args.get("isAsset");
    boolean isAsset = isAssetObj == null ? false : (boolean) isAssetObj;
    MappedByteBuffer buffer = null;
    String key = null;
    AssetManager assetManager = null;
    if (isAsset) {
      assetManager = mRegistrar.context().getAssets();
      key = mRegistrar.lookupKeyForAsset(model);
      AssetFileDescriptor fileDescriptor = assetManager.openFd(key);
      FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    } else {
      FileInputStream inputStream = new FileInputStream(new File(model));
      FileChannel fileChannel = inputStream.getChannel();
      long declaredLength = fileChannel.size();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, declaredLength);
    }

    int numThreads = (int) args.get("numThreads");
    Boolean useGpuDelegate = (Boolean) args.get("useGpuDelegate");
    if (useGpuDelegate == null) {
      useGpuDelegate = false;
    }

    final Interpreter.Options tfliteOptions = new InterpreterApi.Options();
    tfliteOptions.setNumThreads(numThreads);
//    if (useGpuDelegate){
//      GpuDelegate delegate = new GpuDelegate();
//      tfliteOptions.addDelegate(delegate);
//    }
    tfLite = new InterpreterFactory().create(buffer, tfliteOptions);

    return "success";
  }

  Bitmap feedOutput(ByteBuffer imgData, float mean, float std) {
    Tensor tensor = tfLite.getOutputTensor(0);
    int outputSize = tensor.shape()[1];
    Bitmap bitmapRaw = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888);

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 16);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 8);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    } else {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((imgData.get() & 0xFF) << 16);
          pixelValue |= ((imgData.get() & 0xFF) << 8);
          pixelValue |= ((imgData.get() & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    }
    return bitmapRaw;
  }

  ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    int[] shape = tensor.shape();
    inputSize = shape[1];
    int inputChannels = shape[3];

    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = bitmapRaw;
    if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
      Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
          inputSize, inputSize, false);
      bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
      final Canvas canvas = new Canvas(bitmap);
      if (inputChannels == 1){
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        canvas.drawBitmap(bitmapRaw, matrix, paint);
      } else {
        canvas.drawBitmap(bitmapRaw, matrix, null);
      }
    }
    
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = bitmap.getPixel(j, i);
        if (inputChannels > 1){
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else {
          imgData.put((byte) ((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF));
        }
      }
    }

    // if (tensor.dataType() == DataType.FLOAT32) {
    //   for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //       int pixelValue = bitmap.getPixel(j, i);
    //       if (inputChannels > 1){
    //         imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
    //         imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
    //         imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
    //       } else {
    //         imgData.putFloat((((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF) - mean) / std);
    //       }
    //     }
    //   }
    // } else {
    //   for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //       int pixelValue = bitmap.getPixel(j, i);
    //       if (inputChannels > 1){
    //         imgData.put((byte) ((pixelValue >> 16) & 0xFF));
    //         imgData.put((byte) ((pixelValue >> 8) & 0xFF));
    //         imgData.put((byte) (pixelValue & 0xFF));
    //       } else {
    //         imgData.put((byte) ((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF));
    //       }
    //     }
    //   }
    // }

    return imgData;
  }

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  ByteBuffer feedInputTensorFrame(List<byte[]> bytesList, int imageHeight, int imageWidth, float mean, float std, int rotation) throws IOException {
    ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
    ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
    ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));

    int Yb = Y.remaining();
    int Ub = U.remaining();
    int Vb = V.remaining();

    byte[] data = new byte[Yb + Ub + Vb];

    Y.get(data, 0, Yb);
    V.get(data, Yb, Vb);
    U.get(data, Yb + Vb, Ub);

    Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
    Allocation bmData = renderScriptNV21ToRGBA888(
        mRegistrar.context(),
        imageWidth,
        imageHeight,
        data);
    bmData.copyTo(bitmapRaw);

    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  public Allocation renderScriptNV21ToRGBA888(Context context, int width, int height, byte[] nv21) {
    // https://stackoverflow.com/a/36409748
    RenderScript rs = RenderScript.create(context);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
    Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

    Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
    Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

    in.copyFrom(nv21);

    yuvToRgbIntrinsic.setInput(in);
    yuvToRgbIntrinsic.forEach(out);
    return out;
  }

  private abstract class TfliteTask extends AsyncTask<Void, Void, Void> {
    Result result;
    boolean asynch;

    TfliteTask(HashMap args, Result result) {
      if (tfLiteBusy) throw new RuntimeException("Interpreter busy");
      else tfLiteBusy = true;
      Object asynch = args.get("asynch");
      this.asynch = asynch == null ? false : (boolean) asynch;
      this.result = result;
    }

    abstract void runTflite();

    abstract void onRunTfliteDone();

    public void executeTfliteTask() {
      if (asynch) execute();
      else {
        runTflite();
        tfLiteBusy = false;
        onRunTfliteDone();
      }
    }

    protected Void doInBackground(Void... backgroundArguments) {
      runTflite();
      return null;
    }

    protected void onPostExecute(Void backgroundResult) {
      tfLiteBusy = false;
      onRunTfliteDone();
    }
  }

  private class RunModelOnImage extends TfliteTask {
    ByteBuffer input;
    long startTime;
    float[] output;

    RunModelOnImage(HashMap args, Result result) throws IOException {
      super(args, result);

      String path = args.get("path").toString();
      double mean = (double) (args.get("imageMean"));
      float IMAGE_MEAN = (float) mean;
      double std = (double) (args.get("imageStd"));
      float IMAGE_STD = (float) std;

      startTime = SystemClock.uptimeMillis();
      input = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);
    }

    protected void runTflite() {
      tfLite.run(input, output);
    }

    protected void onRunTfliteDone() {
      Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
      result.success(output);
    }
  }

  private class RunModelOnFrame extends TfliteTask {
    long startTime;
    ByteBuffer imgData;
    float[] output;

    RunModelOnFrame(HashMap args, Result result) throws IOException {
      super(args, result);

      ByteBuffer imgData = (ByteBuffer) args.get("byteBuffer");

      startTime = SystemClock.uptimeMillis();
    }

    protected void runTflite() {
      tfLite.run(imgData, output);
    }

    protected void onRunTfliteDone() {
      Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
      result.success(output);
    }
  }

  void setPixel(byte[] rgba, int index, long color) {
    rgba[index * 4] = (byte) ((color >> 16) & 0xFF);
    rgba[index * 4 + 1] = (byte) ((color >> 8) & 0xFF);
    rgba[index * 4 + 2] = (byte) (color & 0xFF);
    rgba[index * 4 + 3] = (byte) ((color >> 24) & 0xFF);
  }

  private static Matrix getTransformationMatrix(final int srcWidth,
                                                final int srcHeight,
                                                final int dstWidth,
                                                final int dstHeight,
                                                final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) srcWidth;
      final float scaleFactorY = dstHeight / (float) srcHeight;

      if (maintainAspectRatio) {
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    matrix.invert(new Matrix());
    return matrix;
  }

  private void close() {
    if (tfLite != null)
      tfLite.close();
      output = null;
  }
}
