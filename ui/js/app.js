/**
 * Auth Guard Controller & Main Application Orchestrator for SatQuery AI
 * Manages 4 Main Feature Cards, Workflow Modes, AI Agent Routing, and Execution.
 */
const AuthGuard = {
  isLoggedIn() {
    return sessionStorage.getItem('satquery_logged_in') === 'true';
  },

  ensureAuthenticated() {
    if (!this.isLoggedIn()) {
      const loginPage = document.getElementById('loginPage');
      if (loginPage) loginPage.classList.remove('hidden');
      return false;
    }
    return true;
  }
};

let activeWorkflowMode = 'AGENT'; // 'AGENT', 'SINGLE', 'CHANGE', 'MULTIMODAL'

document.addEventListener('DOMContentLoaded', () => {
  UploadManager.init();
  ResultsManager.init();

  const loginPage = document.getElementById('loginPage');
  const tabLoginBtn = document.getElementById('tabLoginBtn');
  const tabSignupBtn = document.getElementById('tabSignupBtn');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');
  
  const quickDemoBtn = document.getElementById('quickDemoBtn');
  const logoutBtn = document.getElementById('logoutBtn');

  const analyzeBtn = document.getElementById('analyzeBtn');
  const cancelBtn = document.getElementById('cancelBtn');
  const clearBtn = document.getElementById('clearBtn');
  const queryInput = document.getElementById('queryInput');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const errorBox = document.getElementById('errorBox');

  const agentDetectedTask = document.getElementById('agentDetectedTask');
  const agentRoutingReasonText = document.getElementById('agentRoutingReasonText');
  const agentStatusText = document.getElementById('agentStatusText');
  const ambiguousModal = document.getElementById('ambiguousModal');
  const btnSelectChange = document.getElementById('btnSelectChange');
  const btnSelectMultimodal = document.getElementById('btnSelectMultimodal');

  // Initial Auth Check
  if (AuthGuard.isLoggedIn()) {
    loginPage.classList.add('hidden');
  } else {
    loginPage.classList.remove('hidden');
  }

  // Auth Tabs & Form Handler
  if (tabLoginBtn && tabSignupBtn) {
    tabLoginBtn.addEventListener('click', () => {
      tabLoginBtn.classList.add('active');
      tabSignupBtn.classList.remove('active');
      loginForm.classList.remove('hidden');
      signupForm.classList.add('hidden');
    });

    tabSignupBtn.addEventListener('click', () => {
      tabSignupBtn.classList.add('active');
      tabLoginBtn.classList.remove('active');
      signupForm.classList.remove('hidden');
      loginForm.classList.add('hidden');
    });
  }

  if (loginForm) loginForm.addEventListener('submit', (e) => { e.preventDefault(); unlockAuth(); });
  if (signupForm) signupForm.addEventListener('submit', (e) => { e.preventDefault(); unlockAuth(); });
  if (quickDemoBtn) quickDemoBtn.addEventListener('click', () => unlockAuth());

  function unlockAuth() {
    if (window.heroEngine) {
      window.heroEngine.triggerEnterWarp(() => {
        sessionStorage.setItem('satquery_logged_in', 'true');
        loginPage.classList.add('hidden');
      });
    } else {
      sessionStorage.setItem('satquery_logged_in', 'true');
      loginPage.classList.add('hidden');
    }
  }

  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      sessionStorage.removeItem('satquery_logged_in');
      if (window.heroEngine) window.heroEngine.resetWarp();
      loginPage.classList.remove('hidden');
    });
  }

  // Direct Page/View Switcher (Page 1: Input & Query | Page 2: Dedicated Results Page)
  function switchView(viewName) {
    const pageInput = document.getElementById('pageInput');
    const pageResults = document.getElementById('pageResults');
    const navInputBtn = document.getElementById('navInputBtn');
    const navResultsBtn = document.getElementById('navResultsBtn');
    const dashboardContainer = document.querySelector('.dashboard-header-container');

    if (viewName === 'results') {
      if (pageInput) pageInput.classList.add('hidden');
      if (pageResults) pageResults.classList.remove('hidden');
      if (dashboardContainer) dashboardContainer.classList.add('hidden');

      if (navResultsBtn) {
        navResultsBtn.style.background = 'rgba(34, 197, 94, 0.25)';
        navResultsBtn.style.color = '#4ade80';
        navResultsBtn.style.borderColor = '#4ade80';
      }
      if (navInputBtn) {
        navInputBtn.style.background = 'rgba(255, 255, 255, 0.06)';
        navInputBtn.style.color = '#cbd5e1';
        navInputBtn.style.borderColor = 'var(--border-color)';
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      if (pageResults) pageResults.classList.add('hidden');
      if (pageInput) pageInput.classList.remove('hidden');
      if (dashboardContainer) dashboardContainer.classList.remove('hidden');

      if (navInputBtn) {
        navInputBtn.style.background = 'rgba(56, 189, 248, 0.25)';
        navInputBtn.style.color = '#38bdf8';
        navInputBtn.style.borderColor = '#38bdf8';
      }
      if (navResultsBtn) {
        navResultsBtn.style.background = 'rgba(255, 255, 255, 0.06)';
        navResultsBtn.style.color = '#cbd5e1';
        navResultsBtn.style.borderColor = 'var(--border-color)';
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }
  window.switchView = switchView;

  // Header Direct Navigation Controls
  const navInputBtn = document.getElementById('navInputBtn');
  const navResultsBtn = document.getElementById('navResultsBtn');
  const btnScrollToTop = document.getElementById('btnScrollToTop');
  const btnBackToInputTop = document.getElementById('btnBackToInputTop');

  if (navResultsBtn) {
    navResultsBtn.addEventListener('click', () => {
      if (!AuthGuard.ensureAuthenticated()) return;
      switchView('results');
    });
  }

  if (navInputBtn) navInputBtn.addEventListener('click', () => { if (!AuthGuard.ensureAuthenticated()) return; switchView('input'); });
  if (btnScrollToTop) btnScrollToTop.addEventListener('click', () => { if (!AuthGuard.ensureAuthenticated()) return; switchView('input'); });
  if (btnBackToInputTop) btnBackToInputTop.addEventListener('click', () => { if (!AuthGuard.ensureAuthenticated()) return; switchView('input'); });

  // 1. Dashboard 4 Main Feature Cards Click Handlers
  document.querySelectorAll('.feature-card').forEach((card) => {
    card.addEventListener('click', () => {
      if (!AuthGuard.ensureAuthenticated()) return;
      const mode = card.dataset.mode;
      if (mode === 'IMAGE DETECTION') switchWorkflowMode('SINGLE');
      else if (mode === 'CHANGE ANALYSIS') switchWorkflowMode('CHANGE');
      else if (mode === 'OPTICAL + SAR ANALYSIS') switchWorkflowMode('MULTIMODAL');
      else switchWorkflowMode('AGENT');

      document.getElementById('uploadWorkspace').scrollIntoView({ behavior: 'smooth' });
    });
  });

  // 2. Mode Navigation Tabs Click Handlers
  document.querySelectorAll('.mode-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      if (!AuthGuard.ensureAuthenticated()) return;
      switchWorkflowMode(tab.dataset.workflow);
    });
  });

  function switchWorkflowMode(mode) {
    activeWorkflowMode = mode;
    errorBox.classList.add('hidden');
    ambiguousModal.classList.add('hidden');

    // Update Tab UI
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    const activeTab = document.querySelector(`.mode-tab[data-workflow="${mode}"]`);
    if (activeTab) activeTab.classList.add('active');

    // Update Cards Highlight UI
    document.querySelectorAll('.feature-card').forEach(c => c.classList.remove('active-card'));
    if (mode === 'SINGLE') document.getElementById('cardImageDetection')?.classList.add('active-card');
    else if (mode === 'CHANGE') document.getElementById('cardChangeAnalysis')?.classList.add('active-card');
    else if (mode === 'MULTIMODAL') document.getElementById('cardOpticalSAR')?.classList.add('active-card');
    else document.getElementById('cardAIAgent')?.classList.add('active-card');

    // Hide all mode workspaces then display active workspace
    document.querySelectorAll('.mode-workspace').forEach(w => w.classList.add('hidden'));

    if (mode === 'SINGLE') {
      document.getElementById('workspaceSingle').classList.remove('hidden');
      document.getElementById('workspaceHeading').textContent = 'IMAGE DETECTION — Single Satellite Image';
      updateAgentPanel('IMAGE DETECTION', 'Single Image Detection & VQA workflow selected.');
    } else if (mode === 'CHANGE') {
      document.getElementById('workspaceChange').classList.remove('hidden');
      document.getElementById('workspaceHeading').textContent = 'CHANGE ANALYSIS — Image A (Before) vs Image B (After)';
      updateAgentPanel('CHANGE ANALYSIS', 'Two-Image Change Detection & Change VQA workflow selected.');
    } else if (mode === 'MULTIMODAL') {
      document.getElementById('workspaceMultimodal').classList.remove('hidden');
      document.getElementById('workspaceHeading').textContent = 'OPTICAL + SAR ANALYSIS — Multimodal Joint Capture';
      updateAgentPanel('OPTICAL + SAR ANALYSIS', 'Optical Multispectral + SAR Radar Backscatter Joint Analysis selected.');
    } else {
      document.getElementById('workspaceAgent').classList.remove('hidden');
      document.getElementById('workspaceHeading').textContent = 'AI AGENT — Intelligent Automatic Routing Workspace';
      updateAgentPanel('AI Agent Auto-Routing', 'Upload images and enter a query — the Agent will evaluate count, sensor modality, and query intent.');
    }
  }

  function updateAgentPanel(taskTitle, reasonText) {
    if (agentDetectedTask) agentDetectedTask.textContent = taskTitle;
    if (agentRoutingReasonText) agentRoutingReasonText.textContent = reasonText;
    if (agentStatusText) agentStatusText.textContent = 'Agent Active';
  }

  // 3. Preset Chips & Clear Button
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      if (!AuthGuard.ensureAuthenticated()) return;
      queryInput.value = chip.dataset.query;
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      queryInput.value = '';
      errorBox.classList.add('hidden');
    });
  }

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      API.cancelActiveRequest();
      setLoadingState(false, '🚀 Analyze with SatQuery AI →');
      showError('Analysis request cancelled by user.');
    });
  }

  // Ambiguous Modal Prompt Handlers
  if (btnSelectChange) {
    btnSelectChange.addEventListener('click', () => {
      ambiguousModal.classList.add('hidden');
      switchWorkflowMode('CHANGE');
      triggerAnalysisExecution('CHANGE ANALYSIS');
    });
  }

  if (btnSelectMultimodal) {
    btnSelectMultimodal.addEventListener('click', () => {
      ambiguousModal.classList.add('hidden');
      switchWorkflowMode('MULTIMODAL');
      triggerAnalysisExecution('OPTICAL + SAR ANALYSIS');
    });
  }

  // 4. Main Query Submit & Execution Orchestrator
  analyzeBtn.addEventListener('click', async () => {
    if (!AuthGuard.ensureAuthenticated()) return;
    errorBox.classList.add('hidden');
    ambiguousModal.classList.add('hidden');

    const query = queryInput.value.trim();
    if (!query) {
      showError('Please enter a natural-language question or query.');
      return;
    }

    // Determine inputs based on current mode
    let targetWorkflow = activeWorkflowMode;
    let file1 = null;
    let file2 = null;

    if (activeWorkflowMode === 'SINGLE') {
      file1 = UploadManager.singleFile;
      if (!file1) { showError('Please upload a satellite image for Image Detection.'); return; }
    } else if (activeWorkflowMode === 'CHANGE') {
      file1 = UploadManager.fileA;
      file2 = UploadManager.fileB;
      if (!file1 || !file2) { showError('Please upload both Image A (Before) and Image B (After) for Change Analysis.'); return; }
    } else if (activeWorkflowMode === 'MULTIMODAL') {
      file1 = UploadManager.opticalFile;
      file2 = UploadManager.sarFile;
      if (!file1) { showError('Please upload an Optical Image for Optical + SAR Analysis.'); return; }
    } else {
      // AGENT Mode
      file1 = UploadManager.image1File || UploadManager.singleFile || UploadManager.fileA || UploadManager.opticalFile;
      file2 = UploadManager.image2File || UploadManager.fileB || UploadManager.sarFile;

      if (!file1) {
        // Text-Only Domain Chatbot Call
        setLoadingState(true, 'Consulting SatQuery AI Remote Sensing Domain Specialist...');
        cancelBtn.classList.remove('hidden');
        try {
          const chatRes = await API.chatAgent(query);
          setLoadingState(true, 'Preparing response...');
          ResultsManager.render(chatRes);
          switchView('results');
        } catch (err) {
          showError(err.message || 'Chat query failed.');
        } finally {
          setLoadingState(false, '🚀 Analyze with SatQuery AI →');
          cancelBtn.classList.add('hidden');
        }
        return;
      }

      // Query AI Agent Router
      setLoadingState(true, 'Understanding query & evaluating workflow...');
      const imageCount = file2 ? 2 : 1;
      const routeRes = await API.routeAgent(query, imageCount, null, false, false);

      if (routeRes.task === 'AMBIGUOUS') {
        setLoadingState(false, '🚀 Analyze with SatQuery AI →');
        ambiguousModal.classList.remove('hidden');
        updateAgentPanel('AMBIGUOUS INTENT', routeRes.reason);
        return;
      }

      targetWorkflow = routeRes.task;
      updateAgentPanel(routeRes.task, routeRes.reason);
    }

    await triggerAnalysisExecution(targetWorkflow, file1, file2, query);
  });

  async function triggerAnalysisExecution(targetWorkflow, file1, file2, queryText) {
    const query = queryText || queryInput.value.trim();

    if (!file1) {
      if (activeWorkflowMode === 'SINGLE') file1 = UploadManager.singleFile;
      else if (activeWorkflowMode === 'CHANGE') { file1 = UploadManager.fileA; file2 = UploadManager.fileB; }
      else if (activeWorkflowMode === 'MULTIMODAL') { file1 = UploadManager.opticalFile; file2 = UploadManager.sarFile; }
      else { file1 = UploadManager.image1File; file2 = UploadManager.image2File; }
    }

    setLoadingState(true, 'Preparing image...');
    cancelBtn.classList.remove('hidden');

    try {
      let response = null;

      if (targetWorkflow === 'IMAGE DETECTION' || targetWorkflow === 'SINGLE') {
        setLoadingState(true, 'Processing imagery & understanding query...');
        response = await API.analyzeSingle(file1, query);

      } else if (targetWorkflow === 'CHANGE ANALYSIS' || targetWorkflow === 'CHANGE') {
        const dateA = document.getElementById('dateA')?.value?.trim() || null;
        const locationA = document.getElementById('locationA')?.value?.trim() || null;
        const dateB = document.getElementById('dateB')?.value?.trim() || null;
        const locationB = document.getElementById('locationB')?.value?.trim() || null;

        if (dateA && dateB && dateA >= dateB) {
          throw new Error('Before image must have an earlier acquisition date than After image.');
        }

        setLoadingState(true, 'Running Change Detection & Morphological analysis...');
        response = await API.analyzeChange(file1, file2 || file1, query, dateA, locationA, dateB, locationB);

      } else if (targetWorkflow === 'OPTICAL + SAR ANALYSIS' || targetWorkflow === 'MULTIMODAL') {
        setLoadingState(true, 'Executing Multimodal Optical + SAR Fusion...');
        response = await API.analyzeOpticalSAR(file1, file2, query);

      } else {
        setLoadingState(true, 'Executing SatQuery AI Analysis...');
        response = await API.analyze(file1, file2, query);
      }

      setLoadingState(true, 'Preparing structured results...');
      ResultsManager.render(response);
      switchView('results');

    } catch (err) {
      showError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoadingState(false, '🚀 Analyze with SatQuery AI →');
      cancelBtn.classList.add('hidden');
    }
  }

  function setLoadingState(isLoading, textContent) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      btnSpinner.classList.remove('hidden');
      btnText.textContent = textContent;
    } else {
      analyzeBtn.disabled = false;
      btnSpinner.classList.add('hidden');
      btnText.textContent = textContent;
    }
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
  }

  // Floating Circular Chatbot Widget Controls
  const floatingChatbotToggle = document.getElementById('floatingChatbotToggle');
  const openChatbotBtn = document.getElementById('openChatbotBtn');
  const closeChatbotWidgetBtn = document.getElementById('closeChatbotWidgetBtn');
  const chatbotWidgetPanel = document.getElementById('chatbotWidgetPanel');
  const chatbotWidgetForm = document.getElementById('chatbotWidgetForm');
  const chatbotWidgetInput = document.getElementById('chatbotWidgetInput');
  const chatbotWidgetMessages = document.getElementById('chatbotWidgetMessages');

  function toggleWidgetPanel(show) {
    if (!chatbotWidgetPanel) return;
    const isCurrentlyHidden = chatbotWidgetPanel.classList.contains('hidden');
    const shouldShow = show !== undefined ? show : isCurrentlyHidden;

    if (shouldShow) {
      chatbotWidgetPanel.classList.remove('hidden');
      if (chatbotWidgetInput) chatbotWidgetInput.focus();
    } else {
      chatbotWidgetPanel.classList.add('hidden');
    }
  }

  if (floatingChatbotToggle) {
    floatingChatbotToggle.addEventListener('click', () => toggleWidgetPanel());
  }

  if (openChatbotBtn) {
    openChatbotBtn.addEventListener('click', (e) => {
      e.preventDefault();
      toggleWidgetPanel(true);
    });
  }

  if (closeChatbotWidgetBtn) {
    closeChatbotWidgetBtn.addEventListener('click', () => toggleWidgetPanel(false));
  }

  // Quick Chips inside Chatbot Widget
  document.querySelectorAll('.chatbot-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.dataset.chat;
      if (text && chatbotWidgetInput) {
        chatbotWidgetInput.value = text;
        submitWidgetChatbotQuery(text);
      }
    });
  });

  if (chatbotWidgetForm) {
    chatbotWidgetForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = chatbotWidgetInput.value.trim();
      if (!text) return;
      submitWidgetChatbotQuery(text);
    });
  }

  async function submitWidgetChatbotQuery(queryText) {
    if (!AuthGuard.ensureAuthenticated()) return;
    if (chatbotWidgetInput) chatbotWidgetInput.value = '';

    appendWidgetChatBubble(queryText, 'user');
    const loadingBubble = appendWidgetChatBubble('🛰️ Thinking...', 'bot');

    try {
      const res = await API.chatAgent(queryText);
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.parentNode.removeChild(loadingBubble);
      }

      const isRedirect = res.intent === 'OUT_OF_DOMAIN';
      appendWidgetChatBubble(res.answer, 'bot', isRedirect);
    } catch (err) {
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.parentNode.removeChild(loadingBubble);
      }
      appendWidgetChatBubble(`❌ Error: ${err.message}`, 'bot', true);
    }
  }

  function formatMarkdownHTML(text) {
    if (!text) return '';
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\$\$\\text\{NDVI\}\s*=\s*\\frac\{\\text\{NIR\}\s*-\s*\\text\{Red\}\}\{\\text\{NIR\}\s*\+\s*\\text\{Red\}\}\$\$/gi, '<div style="background: rgba(14,165,233,0.15); border-left: 3px solid #38bdf8; padding: 6px 10px; margin: 6px 0; border-radius: 4px; font-family: monospace; font-weight: 600; color: #7dd3fc;">Formula: NDVI = (NIR - Red) / (NIR + Red)</div>')
      .replace(/\$\$\\text\{NDWI\}\s*=\s*\\frac\{\\text\{Green\}\s*-\s*\\text\{NIR\}\}\{\\text\{Green\}\s*\+\s*\\text\{NIR\}\}\$\$/gi, '<div style="background: rgba(14,165,233,0.15); border-left: 3px solid #38bdf8; padding: 6px 10px; margin: 6px 0; border-radius: 4px; font-family: monospace; font-weight: 600; color: #7dd3fc;">Formula: NDWI = (Green - NIR) / (Green + NIR)</div>')
      .replace(/\$\$\\text\{MNDWI\}\s*=\s*\\frac\{\\text\{Green\}\s*-\s*\\text\{SWIR\}\}\{\\text\{Green\}\s*\+\s*\\text\{SWIR\}\}\$\$/gi, '<div style="background: rgba(14,165,233,0.15); border-left: 3px solid #38bdf8; padding: 6px 10px; margin: 6px 0; border-radius: 4px; font-family: monospace; font-weight: 600; color: #7dd3fc;">Formula: MNDWI = (Green - SWIR) / (Green + SWIR)</div>')
      .replace(/\$\$\\text\{NDBI\}\s*=\s*\\frac\{\\text\{SWIR\}\s*-\s*\\text\{NIR\}\}\{\\text\{SWIR\}\s*\+\s*\\text\{NIR\}\}\$\$/gi, '<div style="background: rgba(14,165,233,0.15); border-left: 3px solid #38bdf8; padding: 6px 10px; margin: 6px 0; border-radius: 4px; font-family: monospace; font-weight: 600; color: #7dd3fc;">Formula: NDBI = (SWIR - NIR) / (SWIR + NIR)</div>')
      .replace(/\$\$(.*?)\$\$/gs, '<div style="background: rgba(14,165,233,0.12); padding: 4px 8px; margin: 4px 0; border-radius: 4px; font-family: monospace; color: #7dd3fc;">$1</div>')
      .replace(/\\text\{(.*?)\}/g, '$1')
      .replace(/\\frac\{(.*?)\}\{(.*?)\}/g, '($1) / ($2)')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.12); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.82rem;">$1</code>')
      .replace(/• (.*?)(?=\n|$)/g, '<div style="margin-left: 8px; margin-top: 3px; display: flex; gap: 6px;"><span>•</span><span>$1</span></div>')
      .replace(/\n/g, '<br>');
    return html;
  }

  function appendWidgetChatBubble(text, sender, isRedirect = false) {
    if (!chatbotWidgetMessages) return null;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble chat-bubble-${sender}` + (isRedirect ? ' chat-bubble-redirect' : '');
    
    if (sender === 'bot') {
      const header = document.createElement('div');
      header.style.display = 'flex';
      header.style.alignItems = 'center';
      header.style.gap = '6px';
      header.style.marginBottom = '6px';
      header.style.fontWeight = '700';
      header.style.fontSize = '0.82rem';
      header.style.color = '#38bdf8';
      
      const img = document.createElement('img');
      img.src = '/assets/chatbot-avatar.png';
      img.alt = 'Bot Avatar';
      img.style.width = '18px';
      img.style.height = '18px';
      img.style.borderRadius = '50%';
      img.style.objectFit = 'cover';
      
      const name = document.createElement('span');
      name.textContent = 'SatQuery AI';
      
      header.appendChild(img);
      header.appendChild(name);
      bubble.appendChild(header);

      const content = document.createElement('div');
      content.innerHTML = formatMarkdownHTML(text);
      bubble.appendChild(content);
    } else {
      bubble.textContent = text;
    }

    chatbotWidgetMessages.appendChild(bubble);
    chatbotWidgetMessages.scrollTop = chatbotWidgetMessages.scrollHeight;
    return bubble;
  }
});

