// DA-AIIMS Service Worker for push notifications
self.addEventListener('push', function(event) {
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'DA-AIIMS Reminder';
  const options = {
    body: data.body || 'Time to take your medication.',
    icon: '/static/icon.png',
    badge: '/static/badge.png',
    tag: 'DA-AIIMS-reminder',
    requireInteraction: true
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', function(event) {
  event.notification.close();
  event.waitUntil(clients.openWindow('/patient/log'));
});
