# DC-GAN Face Inpainting – Bavlibuch Internship Project (2023)

<p align="center">
  <img src="assets/result1.jpg" width="900"/>
  <br>
  <i>Before → Masked → Magically Restored ✨</i>
</p>

Trained a **conditional DC-GAN from scratch** to intelligently fill missing circular regions in high-quality faces (CelebA-HQ dataset streamed directly from AWS S3).

The model reconstructs missing parts photorealistically — perfect for photo restoration, content-aware fill tools, and medical imaging prototypes.

Built end-to-end during my software engineering internship at **Bavlibuch** (healthcare + AI startup).

## 🚀 Live Demo – Try on Your Own Selfie!
```bash
python inference.py --image your_photo.jpg
