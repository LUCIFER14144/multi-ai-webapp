#!/bin/bash
# Git commit script for authentication and UI updates

echo "🔄 Staging authentication and UI files..."
git add app/main.py
git add app/frontend/auth.html
git add app/frontend/dashboard.html
git add app/frontend/dashboard.js
git add requirements.txt

echo "📝 Committing changes..."
git commit -m "feat: Add user authentication and improved dashboard UI

🔐 Authentication System:
- JWT-based user registration and login
- Protected API endpoints with auth middleware  
- User session management with token validation
- Password hashing and secure authentication flow

🎨 Modern Dashboard UI:
- Professional 3-column layout (history, main, tokens)
- User authentication pages with gradient design
- Personal conversation history sidebar
- Enhanced token usage tracking display
- Responsive mobile-friendly interface
- Modern Inter font and improved styling

📊 Enhanced Features:
- Per-user conversation history (last 50 items)
- Real-time token tracking with provider breakdown
- Click history items to reload past conversations
- User profile management with avatar and logout
- Improved form validation and error handling

🔧 Technical Updates:
- Added PyJWT and python-multipart dependencies
- Updated route handlers for auth pages
- Enhanced security with protected endpoints
- Better error handling and user feedback"

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Successfully uploaded authentication system to GitHub!"
echo "🌐 Repository: https://github.com/LUCIFER14144/multi-ai-webapp"