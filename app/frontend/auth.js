// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        // Update button states
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update form visibility
        document.querySelectorAll('.auth-form').forEach(f => f.classList.remove('active'));
        document.getElementById(`${tab}-form`).classList.add('active');
        
        // Clear messages
        hideMessages();
    });
});

// Password strength checker
const registerPassword = document.getElementById('register-password');
const strengthIndicator = document.getElementById('password-strength');
const strengthFill = document.getElementById('strength-fill');
const strengthText = document.getElementById('strength-text');

registerPassword.addEventListener('input', (e) => {
    const password = e.target.value;
    
    if (password.length === 0) {
        strengthIndicator.classList.remove('show');
        return;
    }
    
    strengthIndicator.classList.add('show');
    
    let strength = 0;
    if (password.length >= 6) strength++;
    if (password.length >= 10) strength++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[^a-zA-Z0-9]/.test(password)) strength++;
    
    strengthFill.className = 'strength-fill';
    
    if (strength <= 2) {
        strengthFill.classList.add('strength-weak');
        strengthText.textContent = 'Weak password';
        strengthText.style.color = '#f44336';
    } else if (strength <= 4) {
        strengthFill.classList.add('strength-medium');
        strengthText.textContent = 'Medium strength';
        strengthText.style.color = '#ff9800';
    } else {
        strengthFill.classList.add('strength-strong');
        strengthText.textContent = 'Strong password';
        strengthText.style.color = '#4caf50';
    }
});

// Error and success messages
function showError(message) {
    const errorEl = document.getElementById('error-message');
    errorEl.textContent = message;
    errorEl.classList.add('show');
    
    hideSuccess();
}

function showSuccess(message) {
    const successEl = document.getElementById('success-message');
    successEl.textContent = message;
    successEl.classList.add('show');
    
    hideError();
}

function hideError() {
    document.getElementById('error-message').classList.remove('show');
}

function hideSuccess() {
    document.getElementById('success-message').classList.remove('show');
}

function hideMessages() {
    hideError();
    hideSuccess();
}

// Login form submission
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const submitBtn = e.target.querySelector('.submit-btn');
    const spinner = submitBtn.querySelector('.spinner');
    
    hideMessages();
    
    // Show loading state
    submitBtn.disabled = true;
    spinner.classList.add('show');
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Login failed');
        }
        
        // Store token
        localStorage.setItem('token', data.access_token);
        
        // Show success and redirect
        showSuccess('Login successful! Redirecting...');
        setTimeout(() => {
            window.location.href = '/';
        }, 1000);
        
    } catch (error) {
        showError(error.message);
        submitBtn.disabled = false;
        spinner.classList.remove('show');
    }
});

// Register form submission
document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const submitBtn = e.target.querySelector('.submit-btn');
    const spinner = submitBtn.querySelector('.spinner');
    
    hideMessages();
    
    // Validate password length
    if (password.length < 6) {
        showError('Password must be at least 6 characters long');
        return;
    }
    
    // Show loading state
    submitBtn.disabled = true;
    spinner.classList.add('show');
    
    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Registration failed');
        }
        
        // Store token
        localStorage.setItem('token', data.access_token);
        
        // Show success and redirect
        showSuccess('Account created successfully! Redirecting...');
        setTimeout(() => {
            window.location.href = '/';
        }, 1000);
        
    } catch (error) {
        showError(error.message);
        submitBtn.disabled = false;
        spinner.classList.remove('show');
    }
});

// Check if already logged in
window.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    if (token) {
        // Verify token is valid
        fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => {
            if (response.ok) {
                window.location.href = '/';
            }
        })
        .catch(() => {
            // Invalid token, remove it
            localStorage.removeItem('token');
        });
    }
});
