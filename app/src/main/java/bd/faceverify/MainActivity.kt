package bd.faceverify

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private var interpreter: Interpreter? = null
    private var pendingAction: Int = -1 // 0=register, 1=verify

    private lateinit var nameInput: EditText
    private lateinit var preview: ImageView
    private lateinit var status: TextView
    private lateinit var savedList: TextView

    companion object {
        private const val PICK_IMAGE = 1001
        private const val SIM_THRESHOLD = 0.6f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        nameInput = findViewById(R.id.nameInput)
        preview = findViewById(R.id.preview)
        status = findViewById(R.id.status)
        savedList = findViewById(R.id.savedList)

        findViewById<Button>(R.id.btnRegister).setOnClickListener {
            pendingAction = 0
            pickImage()
        }
        findViewById<Button>(R.id.btnVerify).setOnClickListener {
            pendingAction = 1
            pickImage()
        }

        loadInterpreter()
        refreshList()
    }

    private fun loadInterpreter() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val fd = assets.openFd("mobile_face_net.tflite")
                val channel = fd.createInputStream().channel
                val model: MappedByteBuffer = channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                )
                interpreter = Interpreter(model)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    status.text = "মডেল লোড ব্যর্থ: ${e.message}"
                }
            }
        }
    }

    private fun pickImage() {
        val intent = android.content.Intent(android.content.Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        startActivityForResult(
            android.content.Intent.createChooser(intent, "ছবি বাছাই করুন"),
            PICK_IMAGE
        )
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: android.content.Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PICK_IMAGE || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val action = pendingAction
        status.text = "প্রসেস হচ্ছে..."
        lifecycleScope.launch(Dispatchers.IO) {
            processImage(uri, action)
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        return try {
            val resolver = contentResolver
            // First pass: size
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
            var sample = 1
            var maxDim = maxOf(opts.outWidth, opts.outHeight)
            while (maxDim / sample > 1024) sample *= 2
            val opts2 = BitmapFactory.Options().apply { inSampleSize = sample }
            var bmp = resolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, opts2)
            } ?: return null
            // Fix EXIF rotation
            val rotation = resolver.openInputStream(uri)?.use {
                when (ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                    else -> 0f
                }
            } ?: 0f
            if (rotation != 0f) {
                val m = Matrix()
                m.postRotate(rotation)
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            }
            bmp
        } catch (e: Exception) {
            null
        }
    }

    private suspend fun detectLargestFace(bitmap: Bitmap): Rect? {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()
        val detector = FaceDetection.getClient(options)
        val faces = try {
            detector.process(InputImage.fromBitmap(bitmap, 0)).await()
        } catch (e: Exception) {
            null
        }
        if (faces == null || faces.isEmpty()) return null
        return faces.maxByOrNull { it.boundingBox.width().toLong() * it.boundingBox.height() }!!.boundingBox
    }

    private fun cropFace(bitmap: Bitmap, box: Rect): Bitmap {
        // Expand box by 30%
        val cx = box.centerX().toFloat()
        val cy = box.centerY().toFloat()
        val w = box.width() * 1.3f
        val h = box.height() * 1.5f
        val left = (cx - w / 2).toInt().coerceIn(0, bitmap.width - 1)
        val top = (cy - h / 2).toInt().coerceIn(0, bitmap.height - 1)
        val right = (left + w.toInt()).coerceAtMost(bitmap.width)
        val bottom = (top + h.toInt()).coerceAtMost(bitmap.height)
        val rect = Rect(left, top, right, bottom)
        val w2 = rect.width().coerceAtLeast(1)
        val h2 = rect.height().coerceAtLeast(1)
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, w2, h2, null, false)
    }

    private fun resizeFace(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, 112, 112, true)
    }

    private fun embed(bitmap: Bitmap): FloatArray? {
        val itp = interpreter ?: return null
        val input = ByteBuffer.allocateDirect(112 * 112 * 3 * 4)
        input.order(ByteOrder.nativeOrder())
        val resized = resizeFace(bitmap)
        val px = IntArray(112 * 112)
        resized.getPixels(px, 0, 112, 0, 0, 112, 112)
        for (p in px) {
            input.putFloat(((p shr 16 and 0xFF) / 127.5f) - 1f)
            input.putFloat(((p shr 8 and 0xFF) / 127.5f) - 1f)
            input.putFloat(((p and 0xFF) / 127.5f) - 1f)
        }
        input.rewind()
        val output = Array(1) { FloatArray(192) }
        try {
            itp.run(input, output)
        } catch (e: Exception) {
            return null
        }
        return output[0]
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        val d = (Math.sqrt(na.toDouble()) * Math.sqrt(nb.toDouble())).toFloat()
        return if (d == 0f) 0f else dot / d
    }

    private fun personsFile(): File = File(filesDir, "persons.json")

    private fun loadPersons(): JSONArray {
        val f = personsFile()
        if (!f.exists()) return JSONArray()
        return try { JSONArray(f.readText()) } catch (e: Exception) { JSONArray() }
    }

    private fun savePersons(arr: JSONArray) {
        personsFile().writeText(arr.toString())
    }

    private suspend fun processImage(uri: Uri, action: Int) {
        val bitmap = decodeBitmap(uri)
        if (bitmap == null) {
            showStatus("ছবি পড়া যায়নি")
            return
        }
        val faceBox = detectLargestFace(bitmap)
        if (faceBox == null) {
            showStatus("কোনো মুখ পাওয়া যায়নি। স্পষ্ট ছবি দিন।")
            return
        }
        val face = cropFace(bitmap, faceBox)
        val embedding = embed(face)
        if (embedding == null) {
            showStatus("মুখ থেকে ফিচার বের করা যায়নি")
            return
        }
        withContext(Dispatchers.Main) { preview.setImageBitmap(face) }

        if (action == 0) {
            val name = nameInput.text.toString().trim()
            if (name.isEmpty()) {
                showStatus("আগে গ্রাহকের নাম লিখুন")
                return
            }
            val avatarFile = File(filesDir, "avatar_${System.currentTimeMillis()}.png")
            avatarFile.outputStream().use { face.compress(Bitmap.CompressFormat.PNG, 90, it) }
            val arr = loadPersons()
            val p = JSONObject()
            p.put("name", name)
            p.put("avatar", avatarFile.name)
            p.put("embedding", JSONArray(embedding.toList()))
            p.put("createdAt", System.currentTimeMillis())
            arr.put(p)
            savePersons(arr)
            showStatus("অ্যাভাটার সেভ হয়েছে: $name")
            refreshList()
        } else {
            val arr = loadPersons()
            if (arr.length() == 0) {
                showStatus("আগে অন্তত একজনের অ্যাভাটার সেভ করুন")
                return
            }
            var bestName = ""; var bestSim = -1f
            for (i in 0 until arr.length()) {
                val p = arr.getJSONObject(i)
                val emb = FloatArray(p.getJSONArray("embedding").length())
                for (j in emb.indices) emb[j] = p.getJSONArray("embedding").getDouble(j).toFloat()
                val sim = cosine(embedding, emb)
                if (sim > bestSim) { bestSim = sim; bestName = p.getString("name") }
            }
            val pct = ((bestSim + 1) / 2 * 100).toInt()
            if (bestSim >= SIM_THRESHOLD) {
                showStatus("মিলে গেছে ✔ ব্যক্তি: $bestName (সাদৃশ্য $pct%)")
            } else {
                showStatus("মিলেনি ✘ (সবচেয়ে কাছের: $bestName, সাদৃশ্য $pct%)")
            }
        }
    }

    private fun showStatus(msg: String) {
        runOnUiThread { status.text = msg }
    }

    private fun refreshList() {
        val arr = loadPersons()
        val sb = StringBuilder("সেভ করা অ্যাভাটার (${arr.length()}):\n")
        for (i in 0 until arr.length()) {
            sb.append("• ").append(arr.getJSONObject(i).getString("name")).append("\n")
        }
        savedList.text = sb.toString()
    }
}
