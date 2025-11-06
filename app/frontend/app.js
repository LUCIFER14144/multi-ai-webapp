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
  criticAnalysis: document.getElementById("critic-analysis"),
  metaInfo: document.getElementById("meta-info")
};

let providersData = {};

// Load available providers and models
async function loadProviders() {
  try {
    // Disable generate button until providers are loaded
    elements.generate.disabled = true;
    elements.model.innerHTML = '<option value="">Loading models...</option>';
    
    const response = await fetch("/api/providers");
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    providersData = data.providers;
    updateModelOptions();
    
    // Re-enable generate button after successful load
    elements.generate.disabled = false;
    
  } catch (error) {
    console.error("Failed to load providers:", error);
    elements.model.innerHTML = '<option value="">⚠️ Failed to load models</option>';
    
    // Show error banner with retry option
    showStatus(
      `⚠️ Failed to load AI providers: ${error.message}. Please refresh the page or check if the server is running.`,
      "error"
    );
    
    // Keep generate button disabled
    elements.generate.disabled = true;
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

// Update token tracker display
function updateTokenDisplay(data) {
  // Update total tokens
  const totalTokensEl = document.getElementById("total-tokens");
  if (totalTokensEl && data.total_tokens !== undefined) {
    totalTokensEl.textContent = data.total_tokens.toLocaleString();
  }
  
  // Update prompt type
  const promptTypeEl = document.getElementById("prompt-type-display");
  if (promptTypeEl && data.prompt_type) {
    promptTypeEl.textContent = data.prompt_type;
  }
  
  // Update provider breakdown
  const tokenProvidersEl = document.getElementById("token-providers");
  if (tokenProvidersEl && data.provider_results) {
    tokenProvidersEl.innerHTML = "";
    
    data.provider_results.forEach(result => {
      if (!result.error && result.tokens_used !== undefined) {
        const providerDiv = document.createElement("div");
        providerDiv.className = "provider-token";
        
        const isWinner = result.provider === data.winning_provider;
        const winnerIcon = isWinner ? "👑 " : "";
        
        providerDiv.innerHTML = `
          <div class="provider-name">${winnerIcon}${result.provider}</div>
          <div class="token-details">
            <div class="token-row">
              <span class="token-label">Total:</span>
              <span class="token-value">${result.tokens_used.toLocaleString()}</span>
            </div>
            <div class="token-row">
              <span class="token-label">Prompt:</span>
              <span class="token-value">${result.prompt_tokens?.toLocaleString() || "—"}</span>
            </div>
            <div class="token-row">
              <span class="token-label">Completion:</span>
              <span class="token-value">${result.completion_tokens?.toLocaleString() || "—"}</span>
            </div>
          </div>
        `;
        
        tokenProvidersEl.appendChild(providerDiv);
      }
    });
  }
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
    
    showStatus("🔬 Step 1/3: Researching topic...", "info");
    
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
    
    // Update token tracker
    updateTokenDisplay(data);
    
    // Show results with smooth transition
    elements.results.style.display = "block";
    elements.research.textContent = data.research_notes;
    
    // Show ALL provider results in competition format
    let competitionText = "🏆 AI PROVIDER COMPETITION RESULTS 🏆\n\n";
    data.provider_results.forEach((result, index) => {
      const isWinner = result.provider === data.winning_provider;
      const trophy = isWinner ? "� WINNER " : "";
      
      competitionText += `${trophy}${index + 1}. ${result.provider.toUpperCase()} (${result.model})\n`;
      competitionText += "─".repeat(60) + "\n\n";
      
      if (result.error) {
        competitionText += `❌ ERROR: ${result.error}\n\n`;
      } else {
        competitionText += `${result.answer}\n\n`;
      }
      competitionText += "═".repeat(60) + "\n\n";
    });
    
    elements.drafts.textContent = competitionText;
    
    // Show the merged best answer
    elements.final.textContent = data.final_answer;
    
    // Show the full multi-judge voting report
    elements.criticAnalysis.textContent = data.critic_report;
    
    // Show metadata with badges - highlight the winning provider
    const winnerBadge = `<span class="badge success">👑 ${data.winning_provider.toUpperCase()}</span>`;
    const providerCount = data.provider_results.length;
    const successCount = data.provider_results.filter(r => !r.error).length;
    const modelBadge = `<span class="badge info">${data.winning_model}</span>`;
    elements.metaInfo.innerHTML = `
      <div class="meta-item">
        <span class="meta-label">🏆 Winner (By Vote):</span>
        ${winnerBadge}
      </div>
      <div class="meta-item">
        <span class="meta-label">Model:</span>
        ${modelBadge}
      </div>
      <div class="meta-item">
        <span class="meta-label">Providers Competed:</span>
        <span class="meta-value">${successCount}/${providerCount}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Judges Voted:</span>
        <span class="meta-value">${successCount}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Status:</span>
        <span class="badge success">✓ Voting Complete</span>
      </div>
    `;
    
    showStatus("✅ Multi-judge voting completed successfully!", "success");
    
    // Scroll to results
    setTimeout(() => {
      elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
    
  } catch (error) {
    console.error("Generation error:", error);
    
    let errorMessage = error.message;
    let errorType = "error";
    
    // Provide helpful context for common errors
    if (errorMessage.includes("429") || errorMessage.includes("Rate limit")) {
      errorType = "warning";
      errorMessage = `⏱️ Rate Limit Reached: ${errorMessage}\n\n💡 Solutions:\n` +
                     `• Wait 1-2 minutes before trying again\n` +
                     `• Try a different provider (switch to Gemini or DeepSeek)\n` +
                     `• If using OpenAI free tier, consider upgrading your plan`;
    } else if (errorMessage.includes("401") || errorMessage.includes("Invalid API key")) {
      errorMessage = `🔑 Invalid API Key: ${errorMessage}\n\n💡 Please:\n` +
                     `• Check that you've entered the correct API key\n` +
                     `• Make sure there are no extra spaces\n` +
                     `• Verify the key is active in your provider's dashboard`;
    } else if (errorMessage.includes("403") || errorMessage.includes("forbidden")) {
      errorMessage = `🚫 Access Denied: ${errorMessage}\n\n💡 Your API key may:\n` +
                     `• Not have permission to use this model\n` +
                     `• Be restricted by your organization\n` +
                     `• Need additional setup in the provider's dashboard`;
    } else if (errorMessage.includes("timeout") || errorMessage.includes("timed out")) {
      errorMessage = `⏰ Request Timeout: The AI provider took too long to respond.\n\n💡 Try:\n` +
                     `• Running the request again\n` +
                     `• Using a shorter prompt\n` +
                     `• Switching to a faster model`;
    } else if (errorMessage.includes("connect") || errorMessage.includes("network")) {
      errorMessage = `🌐 Connection Error: ${errorMessage}\n\n💡 Please:\n` +
                     `• Check your internet connection\n` +
                     `• Verify the AI provider's service is available\n` +
                     `• Try again in a moment`;
    }
    
    showStatus(`❌ ${errorMessage}`, errorType);
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
