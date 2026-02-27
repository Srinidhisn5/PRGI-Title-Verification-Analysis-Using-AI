// ROOT CAUSE FIX: Clean, standalone admin JavaScript
// NO inline handlers, NO duplicate logic, NO page reloads

console.log("ADMIN JS LOADED");

document.addEventListener('DOMContentLoaded', function() {
    // Remove all existing inline onclick handlers
    const existingButtons = document.querySelectorAll('button[data-action]');
    existingButtons.forEach(btn => {
        btn.removeAttribute('onclick');
    });

    // Get all approve, reject, and explain buttons
    const approveButtons = document.querySelectorAll('.btn-approve');
    const rejectButtons = document.querySelectorAll('.btn-reject');


    // Attach clean event listeners
    approveButtons.forEach(button => {
        button.addEventListener('click', function() {
            console.log("BUTTON CLICK", this.getAttribute('data-id'), 'approve');
            handleStatusUpdate(this, 'approved');
        });
    });

    rejectButtons.forEach(button => {
        button.addEventListener('click', function() {
            console.log("BUTTON CLICK", this.getAttribute('data-id'), 'reject');
            handleStatusUpdate(this, 'rejected');
        });
    });
});

// ROOT CAUSE FIX: Single function with comprehensive error handling
async function handleStatusUpdate(button, action) {
    const submissionId = parseInt(button.getAttribute('data-id'));
    const row = button.closest('tr');
    const statusBadge = row.querySelector('.status-badge');
    const statusDot = row.querySelector('.status-dot');
    const actionButtonsDiv = row.querySelector('.action-buttons');

    // Validation
    if (!submissionId || isNaN(submissionId)) {
        showNotification('Invalid submission ID', 'error');
        return;
    }

    // Disable button to prevent double-click
    button.disabled = true;
    button.style.opacity = '0.5';

    try {
        const response = await fetch("/admin/update_status", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                id: submissionId,
                status: action
            })
        });

        // CRITICAL: Check if response is OK
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            // Update UI immediately
            updateStatusUI(row, action, statusBadge, statusDot, actionButtonsDiv);
            showNotification(`Successfully ${action}d submission #${submissionId}`, 'success');
        } else {
            throw new Error(data.error || 'Update failed');
        }
    } catch (error) {
        console.error('Status update error:', error);
        // CRITICAL: Rollback UI on error
        button.disabled = false;
        button.style.opacity = '1';
        showNotification(`Error: ${error.message}`, 'error');
    }
}

// ROOT CAUSE FIX: Instant UI updates
function updateStatusUI(row, action, statusBadge, statusDot, actionButtonsDiv) {
    const newStatus = action === 'approved' ? 'Approved' : 'Rejected';
    const newClass = action === 'approved' ? 'status-approved' : 'status-rejected';
    const newDotIcon = action === 'approved' ? '🟢' : '🔴';
    const newDotClass = action === 'approved' ? 'status-dot approved' : 'status-dot rejected';

    // Update status badge
    statusBadge.textContent = newStatus;
    statusBadge.className = `status-badge ${newClass}`;

    // Update status dot
    if (statusDot) {
        statusDot.textContent = newDotIcon;
        statusDot.className = newDotClass;
    }

    // Remove action buttons
    actionButtonsDiv.innerHTML = '';
}

// ROOT CAUSE FIX: Visible error banner system
function showNotification(message, type) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.admin-notification');
    existingNotifications.forEach(n => n.remove());

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `admin-notification notification-${type}`;
    notification.textContent = message;
    
    // Style notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 2rem',
        borderRadius: '8px',
        zIndex: '1000',
        fontWeight: '600',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        transition: 'all 0.3s ease',
        animation: 'fadeIn 0.3s ease'
    });
    
    if (type === 'success') {
        Object.assign(notification.style, {
            background: 'linear-gradient(135deg, #10b981, #059669)',
            color: '#064e3b'
        });
    } else {
        Object.assign(notification.style, {
            background: 'linear-gradient(135deg, #ef4444, #dc2626)',
            color: '#7f1d1d'
        });
    }

    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}
