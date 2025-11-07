// Dashboard JavaScript
class MultiAIDashboard {
  constructor() {
    this.user = null;
    this.token = null;
    this.history = [];
    this.providersData = {};
    
    this.elements = {
      loadingScreen: document.getElementById('loading-screen'),
      appContainer: document.getElementById('app-container'),
      userAvatar: document.getElementById('user-avatar'),
      userName: document.getElementById('user-name'),
      userEmail: document.getElementById('user-email'),
      logoutBtn: document.getElementById('logout-btn'),
      historyList: document.getElementById('history-list'),
      openaiKey: document.getElementById('openai-key'),
      geminiKey: document.getElementById('gemini-key'),
      deepseekKey: document.getElementById('deepseek-key'),
      prompt: document.getElementById('prompt'),
      generate: document.getElementById('generate'),
      btnText: document.getElementById('btn-text'),
      btnSpinner: document.getElementById('btn-spinner'),
      status: document.getElementById('status'),
      results: document.getElementById('results'),
      research: document.getElementById('research'),
      drafts: document.getElementById('drafts'),
      final: document.getElementById('final'),
      criticAnalysis: document.getElementById('critic-analysis'),
      metaInfo: document.getElementById('meta-info'),
      totalTokens: document.getElementById('total-tokens'),
      promptTypeDisplay: document.getElementById('prompt-type-display'),
      tokenProviders: document.getElementById('token-providers')
    };
    
    this.init();
  }
  
  async init() {
    // Check authentication
    this.token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (!this.token || !userData) {
      window.location.href = '/auth.html';
      return;
    }
    
    try {
      this.user = JSON.parse(userData);
      await this.verifyAuth();
      this.setupUI();
      await this.loadHistory();
      this.setupEventListeners();
      this.showApp();
    } catch (error) {
      console.error('Init error:', error);
      this.logout();
    }
  }
  
  async verifyAuth() {
    try {
      const response = await this.apiCall('/api/auth/me', 'GET');
      if (response.ok) {
        this.user = await response.json();
        localStorage.setItem('user', JSON.stringify(this.user));
      } else {
        throw new Error('Authentication failed');
      }
    } catch (error) {
      throw new Error('Auth verification failed');
    }
  }
  
  setupUI() {
    // Set user info
    this.elements.userAvatar.textContent = this.user.username.charAt(0).toUpperCase();
    this.elements.userName.textContent = this.user.username;
    this.elements.userEmail.textContent = this.user.email;
  }
  
  async loadHistory() {
    try {
      const response = await this.apiCall('/api/auth/history', 'GET');
      if (response.ok) {
        this.history = await response.json();
        this.renderHistory();
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  }
  
  renderHistory() {
    if (this.history.length === 0) {
      this.elements.historyList.innerHTML = `
        <div style="text-align: center; color: var(--text-tertiary); font-size: 0.875rem; padding: 2rem;">
          No conversations yet
        </div>
      `;
      return;
    }
    
    this.elements.historyList.innerHTML = this.history.map(item => `
      <div class="history-item" data-id="${item.id}">
        <div class="history-prompt">${this.escapeHtml(item.prompt)}</div>
        <div class="history-meta">
          <div class="history-provider">
            <span class="provider-badge ${item.winning_provider}">${item.winning_provider}</span>
            <span>${item.winning_model}</span>
          </div>
          <div>${this.formatDate(item.timestamp)}</div>
          <div>${item.total_tokens.toLocaleString()} tokens • ${item.prompt_type}</div>
        </div>
      </div>
    `).join('');
    
    // Add click handlers
    this.elements.historyList.querySelectorAll('.history-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.dataset.id;
        const historyItem = this.history.find(h => h.id === id);
        if (historyItem) {
          this.loadHistoryItem(historyItem);
        }
      });
    });
  }
  
  loadHistoryItem(item) {
    // Fill prompt
    this.elements.prompt.value = item.prompt;
    
    // Show stored response (simplified version)
    this.elements.research.textContent = "Historical conversation loaded";
    this.elements.final.textContent = item.response_summary;
    this.elements.criticAnalysis.textContent = `Previous result from ${item.winning_provider} (${item.winning_model})`;
    
    // Update meta info
    this.elements.metaInfo.innerHTML = `
      <div class="meta-item">
        <span class="meta-label">🏆 Previous Winner:</span>
        <span class="badge success">${item.winning_provider.toUpperCase()}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Model:</span>
        <span class="badge info">${item.winning_model}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Tokens Used:</span>
        <span class="meta-value">${item.total_tokens.toLocaleString()}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Date:</span>
        <span class="meta-value">${this.formatDate(item.timestamp)}</span>
      </div>
    `;
    
    // Update token display
    this.elements.totalTokens.textContent = item.total_tokens.toLocaleString();
    this.elements.promptTypeDisplay.textContent = item.prompt_type;
    
    // Show results
    this.elements.results.classList.add('show');
    
    // Scroll to results
    this.elements.results.scrollIntoView({ behavior: 'smooth' });
  }
  
  setupEventListeners() {
    // Logout
    this.elements.logoutBtn.addEventListener('click', () => {
      this.logout();
    });
    
    // Generate button
    this.elements.generate.addEventListener('click', () => {
      this.handleGenerate();
    });
    
    // Enter key in prompt
    this.elements.prompt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey) {
        this.handleGenerate();
      }
    });
  }
  
  async handleGenerate() {
    const prompt = this.elements.prompt.value.trim();
    if (!prompt) {
      this.showStatus('Please enter a prompt.', 'error');
      return;
    }
    
    if (prompt.length < 3) {
      this.showStatus('Prompt must be at least 3 characters long.', 'error');
      return;
    }
    
    if (prompt.length > 2000) {
      this.showStatus('Prompt too long (max 2000 characters).', 'error');
      return;
    }
    
    // Check API keys
    const apiKeys = {
      openai: this.elements.openaiKey.value.trim(),
      gemini: this.elements.geminiKey.value.trim(),
      deepseek: this.elements.deepseekKey.value.trim()
    };
    
    const hasApiKey = Object.values(apiKeys).some(key => key.length > 0);
    if (!hasApiKey) {
      this.showStatus('Please provide at least one API key.', 'warning');
      return;
    }
    
    this.setLoading(true);
    this.hideStatus();
    
    try {
      const response = await this.apiCall('/api/generate', 'POST', {
        prompt,
        api_keys: apiKeys
      });
      
      if (response.ok) {
        const data = await response.json();
        this.displayResults(data);
        await this.loadHistory(); // Refresh history
      } else {
        const error = await response.json();
        this.showStatus(error.detail || 'Generation failed', 'error');
      }
    } catch (error) {
      console.error('Generation error:', error);
      this.showStatus('Network error. Please try again.', 'error');
    } finally {
      this.setLoading(false);
    }
  }
  
  displayResults(data) {
    // Update token tracker
    this.updateTokenDisplay(data);
    
    // Show results
    this.elements.results.classList.add('show');
    this.elements.research.textContent = data.research_notes;
    
    // Show provider competition results
    let competitionText = "🏆 AI PROVIDER COMPETITION RESULTS 🏆\\n\\n";
    data.provider_results.forEach((result, index) => {
      const isWinner = result.provider === data.winning_provider;
      const trophy = isWinner ? "👑 WINNER " : "";
      
      competitionText += `${trophy}${index + 1}. ${result.provider.toUpperCase()} (${result.model})\\n`;
      competitionText += "─".repeat(60) + "\\n\\n";
      
      if (result.error) {
        competitionText += `❌ ERROR: ${result.error}\\n\\n`;
      } else {
        competitionText += `${result.answer}\\n\\n`;
      }
      competitionText += "═".repeat(60) + "\\n\\n";
    });
    
    this.elements.drafts.textContent = competitionText;
    this.elements.final.textContent = data.final_answer;
    this.elements.criticAnalysis.textContent = data.critic_report;
    
    // Show metadata
    const winnerBadge = `<span class="badge success">👑 ${data.winning_provider.toUpperCase()}</span>`;
    const providerCount = data.provider_results.length;
    const successCount = data.provider_results.filter(r => !r.error).length;
    const modelBadge = `<span class="badge info">${data.winning_model}</span>`;
    
    this.elements.metaInfo.innerHTML = `
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
        <span class="meta-label">Total Tokens:</span>
        <span class="meta-value">${data.total_tokens.toLocaleString()}</span>
      </div>
    `;
    
    // Scroll to results
    this.elements.results.scrollIntoView({ behavior: 'smooth' });
  }
  
  updateTokenDisplay(data) {
    // Update total tokens
    if (data.total_tokens !== undefined) {
      this.elements.totalTokens.textContent = data.total_tokens.toLocaleString();
    }
    
    // Update prompt type
    if (data.prompt_type) {
      this.elements.promptTypeDisplay.textContent = data.prompt_type;
    }
    
    // Update provider breakdown
    if (data.provider_results) {
      this.elements.tokenProviders.innerHTML = "";
      
      data.provider_results.forEach(result => {
        if (!result.error && result.tokens_used !== undefined) {
          const providerDiv = document.createElement("div");
          providerDiv.className = `provider-token ${result.provider}`;
          
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
          
          this.elements.tokenProviders.appendChild(providerDiv);
        }
      });
    }
  }
  
  async apiCall(endpoint, method, data = null) {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.token}`
      }
    };
    
    if (data) {
      options.body = JSON.stringify(data);
    }
    
    return fetch(endpoint, options);
  }
  
  setLoading(loading) {
    this.elements.generate.disabled = loading;
    this.elements.btnText.style.display = loading ? 'none' : 'inline';
    this.elements.btnSpinner.style.display = loading ? 'inline-block' : 'none';
  }
  
  showStatus(message, type = 'info') {
    this.elements.status.textContent = message;
    this.elements.status.className = `status-banner show ${type}`;
    
    if (type === 'success') {
      setTimeout(() => {
        this.hideStatus();
      }, 5000);
    }
  }
  
  hideStatus() {
    this.elements.status.classList.remove('show');
  }
  
  showApp() {
    this.elements.loadingScreen.style.display = 'none';
    this.elements.appContainer.classList.add('show');
  }
  
  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/auth.html';
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new MultiAIDashboard();
});