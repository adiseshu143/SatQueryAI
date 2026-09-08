/**
 * API Communication layer for SatQuery AI
 * Provides methods for all 4 workflows & Agent routing with timeout & cancellation support.
 */
const API = {
  baseUrl: '/api',
  activeController: null,

  createAbortSignal(timeoutMs = 60000) {
    if (this.activeController) {
      this.activeController.abort();
    }
    this.activeController = new AbortController();
    const timer = setTimeout(() => {
      if (this.activeController) this.activeController.abort();
    }, timeoutMs);
    return { signal: this.activeController.signal, timer };
  },

  cancelActiveRequest() {
    if (this.activeController) {
      this.activeController.abort();
      this.activeController = null;
    }
  },

  async checkHealth() {
    const res = await fetch(`${this.baseUrl}/health`);
    return await res.json();
  },

  async analyzeSingle(imageFile, queryText) {
    const { signal, timer } = this.createAbortSignal();
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('query', queryText);

      const res = await fetch(`${this.baseUrl}/analyze/single`, {
        method: 'POST',
        body: formData,
        signal
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail || err.message || 'Single Image Detection failed.');
      }
      return await res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Analysis took longer than expected or was cancelled by user.');
      }
      throw err;
    } finally {
      this.activeController = null;
    }
  },

  async analyzeChange(imageAFile, imageBFile, queryText, dateA = null, locationA = null, dateB = null, locationB = null) {
    const { signal, timer } = this.createAbortSignal();
    try {
      const formData = new FormData();
      formData.append('image_a', imageAFile);
      formData.append('image_b', imageBFile);
      formData.append('query', queryText);
      if (dateA) formData.append('date_a', dateA);
      if (locationA) formData.append('location_a', locationA);
      if (dateB) formData.append('date_b', dateB);
      if (locationB) formData.append('location_b', locationB);

      const res = await fetch(`${this.baseUrl}/analyze/change`, {
        method: 'POST',
        body: formData,
        signal
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail || err.message || 'Change Analysis failed.');
      }
      return await res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Analysis took longer than expected or was cancelled by user.');
      }
      throw err;
    } finally {
      this.activeController = null;
    }
  },

  async analyzeOpticalSAR(opticalFile, sarFile, queryText) {
    const { signal, timer } = this.createAbortSignal();
    try {
      const formData = new FormData();
      formData.append('optical_image', opticalFile);
      if (sarFile) {
        formData.append('sar_image', sarFile);
      }
      formData.append('query', queryText);

      const res = await fetch(`${this.baseUrl}/analyze/optical-sar`, {
        method: 'POST',
        body: formData,
        signal
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail || err.message || 'Optical + SAR Analysis failed.');
      }
      return await res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Analysis took longer than expected or was cancelled by user.');
      }
      throw err;
    } finally {
      this.activeController = null;
    }
  },

  async routeAgent(queryText, imageCount = 1, modeSelected = null, hasSAR = false, hasOptical = false) {
    const formData = new FormData();
    formData.append('query', queryText);
    formData.append('image_count', imageCount);
    if (modeSelected) formData.append('mode_selected', modeSelected);
    formData.append('has_sar', hasSAR);
    formData.append('has_optical', hasOptical);

    const res = await fetch(`${this.baseUrl}/agent/route`, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      return { task: 'IMAGE DETECTION', reason: 'Defaulting to Image Detection.', status: 'routed' };
    }
    return await res.json();
  },

  async analyze(image1File, image2File, queryText) {
    const { signal, timer } = this.createAbortSignal();
    try {
      const formData = new FormData();
      formData.append('image1', image1File);
      if (image2File) {
        formData.append('image2', image2File);
      }
      formData.append('query', queryText);

      const res = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        body: formData,
        signal
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ message: 'Server error' }));
        throw new Error(err.message || 'Failed to perform analysis');
      }

      return await res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Analysis took longer than expected or was cancelled by user.');
      }
      throw err;
    } finally {
      this.activeController = null;
    }
  },

  async chatAgent(queryText) {
    const { signal, timer } = this.createAbortSignal();
    try {
      const formData = new FormData();
      formData.append('query', queryText);

      const res = await fetch(`${this.baseUrl}/agent/chat`, {
        method: 'POST',
        body: formData,
        signal
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(err.detail || err.message || 'Chat request failed.');
      }

      return await res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Request took longer than expected or was cancelled by user.');
      }
      throw err;
    } finally {
      this.activeController = null;
    }
  }
};
