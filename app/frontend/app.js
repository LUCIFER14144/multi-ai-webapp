// DOM elements
const elements = {
  generate: document.getElementById("generate"),
  btnText: document.getElementById("btn-text"),
  btnSpinner: document.getElementById("btn-spinner"),
  prompt: document.getElementById("prompt"),
  provider: document.getElementById("provider"),
  model: document.getElementById("model"),
  openaiKey: document.getElementById("openai-key"),
  geminiKey: document.getElementById("gemini-key"),
  deepseekKey: document.getElementById("deepseek-key"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  research: document.getElementById("research"),
  drafts: document.getElementById("drafts"),
  final: document.getElementById("final"),
  metaInfo: document.getElementById("meta-info")
};

let providersData = {};

// Load available providers and models
async function loadProviders() {
  try {
    const response = await fetch("/api/providers");
    const data = await response.json();
    providersData = data.providers;
    updateModelOptions();
  } catch (error) {
    console.error("Failed to load providers:", error);
    elements.model.innerHTML = '<option value="">Error loading models</option>';
  }
}

// Update model dropdown based on selected provider
function updateModelOptions() {
  const selectedProvider = elements.provider.value;
  const providerConfig = providersData[selectedProvider];
  
  if (!providerConfig) {
    elements.model.innerHTML = '<option value="">No models available</option>';
    return;
  }
  
  elements.model.innerHTML = '';
  providerConfig.models.forEach(model => {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    if (model === providerConfig.default_model) {
      option.selected = true;
    }
    elements.model.appendChild(option);
  });
}

// Validate inputs
function validateInputs() {
  const prompt = elements.prompt.value.trim();
  const provider = elements.provider.value;
  
  if (!prompt) {
    showStatus("Please enter a prompt.", "error");
    return false;
  }
  
  if (prompt.length < 3) {
    showStatus("Prompt must be at least 3 characters long.", "error");
    return false;
  }
  
  if (prompt.length > 2000) {
    showStatus("Prompt too long (max 2000 characters).", "error");
    return false;
  }
  
  // Check if required API key is provided
  let apiKey = null;
  if (provider === "openai") {
    apiKey = elements.openaiKey.value.trim();
  } else if (provider === "gemini") {
    apiKey = elements.geminiKey.value.trim();
  } else if (provider === "deepseek") {
    apiKey = elements.deepseekKey.value.trim();
  }
  
  if (!apiKey) {
    showStatus(`⚠️ Please provide an API key for ${provider.toUpperCase()}.`, "warning");
    return false;
  }
  
  if (apiKey.length < 10) {
    showStatus("⚠️ API key appears to be invalid (too short).", "warning");
    return false;
  }
  
  return true;
}

// Show status message with modern styling
function showStatus(message, type = "info") {
  elements.status.textContent = message;
  elements.status.className = `status-banner show ${type}`;
  
  // Auto-hide success messages after 5 seconds
  if (type === "success") {
    setTimeout(() => {
      elements.status.classList.remove("show");
    }, 5000);
  }
}

// Hide status
function hideStatus() {
  elements.status.classList.remove("show");
}

// Build API request
function buildRequest() {
  const provider = elements.provider.value;
  const model = elements.model.value;
  
  return {
    prompt: elements.prompt.value.trim(),
    provider: provider,
    model: model || null,
    api_keys: {
      openai: elements.openaiKey.value.trim() || null,
      gemini: elements.geminiKey.value.trim() || null,
      deepseek: elements.deepseekKey.value.trim() || null
    }
  };
}

// Set loading state
function setLoading(loading) {
  elements.generate.disabled = loading;
  
  if (loading) {
    elements.btnText.style.display = "none";
    elements.btnSpinner.style.display = "inline-block";
  } else {
    elements.btnText.style.display = "inline";
    elements.btnSpinner.style.display = "none";
  }
}

// Main generate function
async function generate() {
  if (!validateInputs()) return;
  
  // Set loading state
  setLoading(true);
  elements.results.style.display = "none";
  
  showStatus("🚀 Initializing AI pipeline...", "info");
  
  try {
    const requestData = buildRequest();
    
    showStatus("🔬 Running Researcher → Writer → Critic pipeline...", "info");
    
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || response.statusText);
    }
    
    const data = await response.json();
    
    // Show results with smooth transition
    elements.results.style.display = "block";
    elements.research.textContent = data.research_notes;
    elements.drafts.textContent = data.drafts.join("\n\n═══════════ VARIANT SEPARATOR ═══════════\n\n");
    elements.final.textContent = data.final_answer;
    
    // Show metadata with badges
    const providerBadge = `<span class="badge success">${data.provider_used.toUpperCase()}</span>`;
    elements.metaInfo.innerHTML = `
      <div class="meta-item">
        <span class="meta-label">Provider:</span>
        ${providerBadge}
      </div>
      <div class="meta-item">
        <span class="meta-label">Model:</span>
        <span class="meta-value">${data.model_used}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Draft Variants:</span>
        <span class="meta-value">${data.drafts.length}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Status:</span>
        <span class="badge success">✓ Completed</span>
      </div>
    `;
    
    showStatus("✅ Pipeline completed successfully!", "success");
    
    // Scroll to results
    setTimeout(() => {
      elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
    
  } catch (error) {
    console.error("Generation error:", error);
    showStatus(`❌ Error: ${error.message}`, "error");
    elements.results.style.display = "none";
  } finally {
    setLoading(false);
  }
}

// Event listeners
elements.provider.addEventListener("change", updateModelOptions);
elements.generate.addEventListener("click", generate);

// Handle Enter key in prompt textarea (Ctrl+Enter to submit)
elements.prompt.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") {
    generate();
  }
});

// Add input event listeners for real-time validation feedback
[elements.openaiKey, elements.geminiKey, elements.deepseekKey].forEach(input => {
  input.addEventListener("input", () => {
    if (input.value.length > 0 && input.value.length < 10) {
      input.style.borderColor = "#f59e0b";
    } else if (input.value.length >= 10) {
      input.style.borderColor = "#10b981";
    } else {
      input.style.borderColor = "#e5e7eb";
    }
  });
});

// Initialize
loadProviders();
