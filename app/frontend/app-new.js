// Authentication check
let currentUser = null;
let chatHistory = [];

async function checkAuth() {
    const token = localStorage.getItem('token');
    
    if (!token) {
        window.location.href = '/static/auth.html';
        return false;
    }
    
    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Unauthorized');
        }
        
        currentUser = await response.json();
        document.getElementById('user-name').textContent = currentUser.username;
        
        // Load chat history
        await loadChatHistory();
        
        return true;
    } catch (error) {
        localStorage.removeItem('token');
        window.location.href = '/static/auth.html';
        return false;
    }
}

// Load chat history
async function loadChatHistory() {
    const token = localStorage.getItem('token');
    
    try {
        const response = await fetch('/api/history', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) throw new Error('Failed to load history');
        
        chatHistory = await response.json();
        renderChatHistory();
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('chat-history').innerHTML = '<div class="loading">Failed to load history</div>';
    }
}

// Render chat history
function renderChatHistory() {
    const container = document.getElementById('chat-history');
    
    if (chatHistory.length === 0) {
        container.innerHTML = '<div class="loading">No chat history yet</div>';
        return;
    }
    
    container.innerHTML = chatHistory.map((chat, index) => `
        <div class="history-item" data-id="${chat.id}">
            <div class="history-prompt">${truncate(chat.prompt, 50)}</div>
            <div class="history-meta">
                <span>${chat.winning_provider || 'N/A'}</span>
                <span>${formatDate(chat.created_at)}</span>
            </div>
        </div>
    `).join('');
    
    // Add click handlers
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => loadChatFromHistory(parseInt(item.dataset.id)));
    });
}

// Load specific chat from history
function loadChatFromHistory(chatId) {
    const chat = chatHistory.find(c => c.id === chatId);
    if (!chat) return;
    
    // Display the historical result
    const resultsArea = document.getElementById('results-area');
    resultsArea.innerHTML = `
        <div class="result-card">
            <div class="result-header">
                <span class="result-title">📝 Original Prompt</span>
            </div>
            <div class="result-content">${chat.prompt}</div>
        </div>
        
        <div class="result-card">
            <div class="result-header">
                <span class="result-title">🏆 Best Answer</span>
                <span class="winner-badge">${chat.winning_provider} (${chat.winning_model})</span>
            </div>
            <div class="result-content">${chat.final_answer || 'No answer available'}</div>
        </div>
    `;
    
    // Update token display
    document.getElementById('total-tokens').textContent = chat.total_tokens.toLocaleString();
    document.getElementById('prompt-type-display').textContent = chat.prompt_type || '—';
    
    // Highlight active chat
    document.querySelectorAll('.history-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.id) === chatId);
    });
}

// New chat
document.getElementById('new-chat-btn').addEventListener('click', () => {
    document.getElementById('results-area').innerHTML = '';
    document.getElementById('prompt').value = '';
    document.getElementById('total-tokens').textContent = '0';
    document.getElementById('prompt-type-display').textContent = '—';
    document.getElementById('token-providers').innerHTML = '';
    document.querySelectorAll('.history-item').forEach(item => item.classList.remove('active'));
});

// Logout
document.getElementById('logout-btn').addEventListener('click', () => {
    localStorage.removeItem('token');
    window.location.href = '/static/auth.html';
});

// Generate button
document.getElementById('generate-btn').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value.trim();
    const openaiKey = document.getElementById('openai-key').value.trim();
    const geminiKey = document.getElementById('gemini-key').value.trim();
    const deepseekKey = document.getElementById('deepseek-key').value.trim();
    
    if (!prompt) {
        showStatus('Please enter a prompt', 'error');
        return;
    }
    
    if (!openaiKey && !geminiKey && !deepseekKey) {
        showStatus('Please provide at least one API key', 'error');
        return;
    }
    
    const token = localStorage.getItem('token');
    const btn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    
    // Show loading state
    btn.disabled = true;
    btnText.textContent = 'Generating...';
    btnSpinner.style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                prompt,
                api_keys: {
                    openai: openaiKey || null,
                    gemini: geminiKey || null,
                    deepseek: deepseekKey || null
                }
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Generation failed');
        }
        
        // Display results
        displayResults(data);
        
        // Reload history
        await loadChatHistory();
        
        showStatus('✅ Generation complete!', 'success');
        
    } catch (error) {
        showStatus(`❌ ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btnText.textContent = '🚀 Generate';
        btnSpinner.style.display = 'none';
    }
});

// Display results
function displayResults(data) {
    const resultsArea = document.getElementById('results-area');
    
    let html = `
        <div class="result-card">
            <div class="result-header">
                <span class="result-title">📝 Your Prompt</span>
            </div>
            <div class="result-content">${data.research_notes}</div>
        </div>
        
        <div class="result-card">
            <div class="result-header">
                <span class="result-title">🏆 Best Answer</span>
                <span class="winner-badge">${data.winning_provider} - ${data.winning_model}</span>
            </div>
            <div class="result-content">${data.final_answer}</div>
        </div>
    `;
    
    // Show all provider results
    if (data.provider_results && data.provider_results.length > 0) {
        html += '<div class="result-card"><div class="result-header"><span class="result-title">🤖 All AI Responses</span></div>';
        
        data.provider_results.forEach((result, index) => {
            const isWinner = result.provider === data.winning_provider;
            html += `
                <div style="margin-bottom: 20px; padding: 16px; background: ${isWinner ? '#f0fdf4' : '#f9fafb'}; border-radius: 8px;">
                    <div style="font-weight: 700; margin-bottom: 8px; color: ${isWinner ? '#16a34a' : '#374151'};">
                        ${isWinner ? '👑 ' : ''}${index + 1}. ${result.provider.toUpperCase()} (${result.model})
                    </div>
                    ${result.error ? `<div style="color: #ef4444;">❌ ${result.error}</div>` : `<div style="white-space: pre-wrap;">${result.answer}</div>`}
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    // Show critic report
    if (data.critic_report) {
        html += `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">⚖️ Judge Analysis</span>
                </div>
                <div class="result-content">${data.critic_report}</div>
            </div>
        `;
    }
    
    resultsArea.innerHTML = html;
    resultsArea.scrollTop = 0;
    
    // Update token display
    updateTokenDisplay(data);
}

// Update token display
function updateTokenDisplay(data) {
    document.getElementById('total-tokens').textContent = (data.total_tokens || 0).toLocaleString();
    document.getElementById('prompt-type-display').textContent = data.prompt_type || '—';
    
    const tokenProvidersEl = document.getElementById('token-providers');
    tokenProvidersEl.innerHTML = '';
    
    if (data.provider_results) {
        data.provider_results.forEach(result => {
            if (!result.error && result.tokens_used) {
                const providerDiv = document.createElement('div');
                providerDiv.className = 'provider-token';
                
                const isWinner = result.provider === data.winning_provider;
                
                providerDiv.innerHTML = `
                    <div class="provider-name">${isWinner ? '👑 ' : ''}${result.provider}</div>
                    <div class="token-row">
                        <span class="token-label">Total:</span>
                        <span class="token-value">${result.tokens_used.toLocaleString()}</span>
                    </div>
                    <div class="token-row">
                        <span class="token-label">Prompt:</span>
                        <span class="token-value">${result.prompt_tokens?.toLocaleString() || '—'}</span>
                    </div>
                    <div class="token-row">
                        <span class="token-label">Completion:</span>
                        <span class="token-value">${result.completion_tokens?.toLocaleString() || '—'}</span>
                    </div>
                `;
                
                tokenProvidersEl.appendChild(providerDiv);
            }
        });
    }
}

// Show status message
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status-banner show ${type}`;
    
    setTimeout(() => {
        statusEl.classList.remove('show');
    }, 5000);
}

// Utility functions
function truncate(str, length) {
    return str.length > length ? str.substring(0, length) + '...' : str;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
}

// Initialize app
window.addEventListener('DOMContentLoaded', async () => {
    await checkAuth();
});
