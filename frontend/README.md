# TrackShift Frontend

Energy & Overtake Intelligence dashboard — React + Vite, glassmorphism UI.

## Setup

```bash
npm install
npm run dev
```

Then open the printed local URL (defaults to http://localhost:5173).

## Build for production

```bash
npm run build
npm run preview
```

## Project structure

```
trackshift-frontend/
├── src/
│   ├── App.jsx                          # Renders the dashboard
│   ├── main.jsx                         # React entry point
│   ├── index.css                        # Base reset / background color
│   └── components/
│       └── TrackShiftGlassDashboard.jsx # Main dashboard component
├── index.html
├── package.json
└── vite.config.js
```

## Swapping in real data

All race/energy data lives in the `RACE_DATA` object near the top of
`src/components/TrackShiftGlassDashboard.jsx`. Replace it with your live
API response (same shape) to wire this up to your backend — no other
changes needed, since every section only renders whatever is in that object.

## Notes

- The background photo is currently embedded as a base64 string directly in
  the component file (`RACE_BG_IMAGE` constant). Swap that string for your
  own licensed image, or point it at a hosted image URL instead, whenever
  you're ready to replace it.
- Charts use `recharts`; icons use `lucide-react`. Both are declared in
  `package.json`.
