// Minimal service worker: enables "Add to Home Screen" install on Android/iOS
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {}); // network-first (API responses must stay fresh)
