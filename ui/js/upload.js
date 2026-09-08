/**
 * Drag & Drop, Image Validation and Preview Controller with Auth Guard
 */
const UploadManager = {
  // Mode-specific file state containers
  singleFile: null,
  fileA: null,
  fileB: null,
  opticalFile: null,
  sarFile: null,
  image1File: null,
  image2File: null,

  init() {
    // Mode Single (Image Detection)
    this.setupGenericDropzone('dropzoneSingle', 'fileInputSingle', (file) => {
      this.singleFile = file;
    }, () => {
      this.singleFile = null;
    }, '📡', 'Satellite Image (Single Image VQA)', 'JPG, PNG, TIFF, GeoTIFF (max 50MB)');

    // Mode Change (Change Analysis)
    this.setupGenericDropzone('dropzoneA', 'fileInputA', (file) => {
      this.fileA = file;
    }, () => {
      this.fileA = null;
    }, '📅', 'IMAGE A — Earlier / Before', 'Earlier Satellite Capture');

    this.setupGenericDropzone('dropzoneB', 'fileInputB', (file) => {
      this.fileB = file;
    }, () => {
      this.fileB = null;
    }, '📆', 'IMAGE B — Later / After', 'Later Satellite Capture');

    // Mode Multimodal (Optical + SAR)
    this.setupGenericDropzone('dropzoneOptical', 'fileInputOptical', (file) => {
      this.opticalFile = file;
    }, () => {
      this.opticalFile = null;
    }, '🌱', 'OPTICAL IMAGE', 'Multispectral RGB / Sentinel-2');

    this.setupGenericDropzone('dropzoneSAR', 'fileInputSAR', (file) => {
      this.sarFile = file;
    }, () => {
      this.sarFile = null;
    }, '📡', 'SAR IMAGE', 'Sentinel-1 Radar Backscatter (VV/VH)');

    // Mode Agent (Default Flex Grid)
    this.setupGenericDropzone('dropzone1', 'fileInput1', (file) => {
      this.image1File = file;
    }, () => {
      this.image1File = null;
    }, '📡', 'Satellite Image 1 (Required)', 'Drag & drop or browse');

    this.setupGenericDropzone('dropzone2', 'fileInput2', (file) => {
      this.image2File = file;
    }, () => {
      this.image2File = null;
    }, '🛰️', 'Satellite Image 2 (Optional)', 'Before / After or SAR image');
  },

  setupGenericDropzone(zoneId, inputId, setFileCallback, clearFileCallback, iconEmoji, titleText, subtitleText) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    let input = document.getElementById(inputId);

    zone.addEventListener('click', (e) => {
      if (e.target.classList.contains('remove-btn')) return;
      if (!AuthGuard.ensureAuthenticated()) return;
      if (input) input.click();
    });

    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (!AuthGuard.ensureAuthenticated()) return;
      if (e.dataTransfer.files.length > 0) {
        this.processUploadedFile(e.dataTransfer.files[0], zone, inputId, setFileCallback, clearFileCallback, iconEmoji, titleText, subtitleText);
      }
    });

    if (input) {
      input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.processUploadedFile(e.target.files[0], zone, inputId, setFileCallback, clearFileCallback, iconEmoji, titleText, subtitleText);
        }
      });
    }
  },

  processUploadedFile(file, zone, inputId, setFileCallback, clearFileCallback, iconEmoji, titleText, subtitleText) {
    // Validate file size (max 50MB)
    const maxBytes = 50 * 1024 * 1024;
    if (file.size > maxBytes) {
      alert(`File size (${roundMb(file.size)} MB) exceeds maximum allowed size of 50 MB.`);
      return;
    }

    // Validate format
    const nameLower = file.name.lowerCase ? file.name.lowerCase() : file.name.toLowerCase();
    const validExts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'];
    const hasValidExt = validExts.some(ext => nameLower.endsWith(ext));

    if (!hasValidExt) {
      alert(`Unsupported image format. Please upload a JPG, JPEG, PNG, TIFF, or GeoTIFF image file.`);
      return;
    }

    setFileCallback(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      zone.innerHTML = `
        <img src="${e.target.result}" class="preview-img" alt="${titleText}">
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">${file.name} (${roundMb(file.size)} MB)</div>
        <button class="remove-btn" title="Remove file">✕</button>
      `;
      const removeBtn = zone.querySelector('.remove-btn');
      if (removeBtn) {
        removeBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          clearFileCallback();
          this.resetDropzoneMarkup(zone, inputId, iconEmoji, titleText, subtitleText, setFileCallback, clearFileCallback);
        });
      }
    };
    reader.readAsDataURL(file);
  },

  resetDropzoneMarkup(zone, inputId, iconEmoji, titleText, subtitleText, setFileCallback, clearFileCallback) {
    zone.innerHTML = `
      <div style="font-size: 24px;">${iconEmoji}</div>
      <p><strong>${titleText}</strong></p>
      <p>${subtitleText}</p>
      <input type="file" id="${inputId}" accept="image/*,.tif,.tiff" class="hidden">
    `;
    const newInput = document.getElementById(inputId);
    if (newInput) {
      newInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.processUploadedFile(e.target.files[0], zone, inputId, setFileCallback, clearFileCallback, iconEmoji, titleText, subtitleText);
        }
      });
    }
  }
};

function roundMb(bytes) {
  return (bytes / (1024 * 1024)).toFixed(1);
}
