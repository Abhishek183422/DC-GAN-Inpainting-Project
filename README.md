# DC-GAN Face Inpainting – Bavlibuch Internship Project (2023)

# https://docs.google.com/document/d/1TEOyqo1F6ijlHLJx9M9Cc4JqmvDi76UbPOT1iKSX-kw/edit?usp=sharing

Trained a **conditional DC-GAN from scratch** to intelligently fill missing circular regions in high-quality faces (CelebA-HQ dataset streamed directly from AWS S3).

The model reconstructs missing parts photorealistically — perfect for photo restoration, content-aware fill tools, and medical imaging prototypes.

Built end-to-end during my software engineering internship at **Bavlibuch** (healthcare + AI startup).

## 🚀 Live Demo – Try on Your Own Selfie!
```bash
python inference.py --image your_photo.jpg
