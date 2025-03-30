# 🏥 Diagnosis Agent
**Diagnosis Agent** is an AI-powered medical assistant that processes medical reports, highlights abnormalities, and provides simplified summaries for easy understanding. It supports multiple input formats, including **PDFs, medical images (X-rays, MRIs), and textual reports**. The agent is designed to help patients, doctors, and researchers by extracting critical information from medical documents.

---

### Features

>📝 **Medical Report Analysis**  
>Extracts key information from medical reports, including diagnosis, test results, and observations.

>📄 **PDF & Image Processing**  
>Supports **PDFs, X-rays, MRIs, and scanned medical documents** for comprehensive analysis.

>🧪 **Abnormality Detection**  
>Identifies **out-of-range values** in test results and highlights potential medical concerns.

>📊 **Summarization**  
>Generates **easy-to-understand summaries** of complex medical reports.

>🖼️ **Medical Image Interpretation**  
>Uses **AI-powered analysis** to detect anomalies in **X-rays, MRIs, and CT scans**.

>💡 **Medical Data Explanation**  
>Explains **medical terms and test parameters** with simple, user-friendly definitions.

>☁️ **Cloud Deployment (Planned)**  
>Deploying on **Google Cloud Run** for accessibility via web and mobile applications.

>📱 **Mobile App (Planned)**  
>A dedicated **Android & iOS app** for easy report analysis on the go.


### Agents Overview

Below is a summary of the specialized medical diagnosis agents implemented so far, including the medical image types each handles and the AI models used:

| Agent Name            | Medical Image Type | Model Used                                                                     |
|-----------------------|--------------------|--------------------------------------------------------------------------------|
| **ChestXrayAgent**    | Chest X-ray        | **CheXNet (DenseNet-121)**                                                     |
| **BrainMRIAgent**     | Brain MRI          | **VGG-16 CNN (trained from scratch)**                                          |
| **LungCancerAgent**   | Lung CT Scan       | **ResNet-18 CNN (trained from scratch)**                                       |
| **ReportSummarizerAgent** | Text Reports       | **Text Extraction (pdfplumber & Tesseract OCR) and GPT-3.5 (LLM Integration)** |
| **ReportHandlerAgent** | Report Management  | **Handles the communication between all the agents**                           |
