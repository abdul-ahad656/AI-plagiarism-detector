echo "🛠 Starting deployment setup..."

set -e

echo "📦 Updating pip..."
pip install --upgrade pip


echo "📥 Installing backend dependencies..."
pip install -r backend/requirements.txt


cd backend


echo "🚀 Starting Flask backend..."
exec gunicorn app:app --bind 0.0.0.0:$PORT
