
# 🎬 Movies Web App Using TMDb API

This is a responsive web application built using **React.js** and **Tailwind CSS**, powered by the [TMDb API](https://api.themoviedb.org/) to fetch real-time movie data. Users can search for any movie and instantly view its details including the poster, plot, rating, genre, and more.

---

## ✨ Features

- 🔍 Search movies by title in real time
- 🖼️ View high-quality movie posters
- 🎥 Get full details: title, genre, rating, actors, plot, and more
- 💡 Smooth user experience with a dark-themed interface
- 📱 Fully responsive design using Tailwind CSS

---

## 🛠️ Built With

- **React.js** – frontend JavaScript framework
- **Tailwind CSS** – utility-first CSS framework
- **TMDb API** – movie data source

---

## 📦 Project Structure

```
📂 Movies-Webapp-TMDB
├── public/
├── src/
│   ├── components/
│   ├── App.jsx
│   ├── index.js
├── tailwind.config.js
├── package.json
├── README.md        # This file
```

---

## 🚀 Getting Started

Follow these steps to get a local copy running:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movies-webapp-tmdb.git
cd movies-webapp-tmdb
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Get a TMDb API Key

- Go to [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)
- Sign up for a free API key

### 4. Add API Key

Create a `.env` file in the root directory and add:

```
VITE_TMDB_API_KEY=your_api_key_here
```

### 5. Start the App

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to view it in the browser.

---

## 📸 Screenshots

_You can add these files in a `screenshots/` folder and embed them below._

| Search Page         | Movie Details        |
|---------------------|----------------------|
| ![](screenshots/search.png) | ![](screenshots/details.png) |

---

## 💡 Future Enhancements

- Add search suggestions or auto-complete
- Add favorite movie saving using `localStorage`
- Support series and episode listings
- Add trailer links from YouTube
- Add light/dark theme toggle

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Feedback and ideas are welcome!

---

Enjoy discovering movies! 🍿
