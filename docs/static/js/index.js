window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Initialize audio timeline functionality
    initializeAudioTimeline();
})

// Audio Timeline Functionality
function initializeAudioTimeline() {
    // Get all audio elements in the multi-model comparison section
    const audioElements = document.querySelectorAll('.comparison-audio');
    const timelineMarkers = document.querySelectorAll('.timeline-marker');
    const timelineTrack = document.querySelector('.timeline-track');
    const timelineProgress = document.getElementById('timelineProgress');
    
    if (!audioElements.length || !timelineMarkers.length) {
        console.log('Audio timeline elements not found');
        return; // Exit if elements not found
    }

    console.log(`Found ${audioElements.length} audio elements for timeline control`);

    // Add click event listeners to timeline markers
    timelineMarkers.forEach(marker => {
        marker.addEventListener('click', function(e) {
            // Prevent event bubbling to timeline track
            e.stopPropagation();
            
            const targetTime = parseFloat(this.getAttribute('data-time'));
            jumpToTime(targetTime);
            
            // Visual feedback
            this.classList.add('active');
            setTimeout(() => {
                this.classList.remove('active');
            }, 400);
        });
    });

    // Add click event to timeline track for scrubbing
    timelineTrack.addEventListener('click', function(e) {
        const rect = this.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = clickX / rect.width;
        const maxDuration = 93; // 1:33 = 93 seconds
        const targetTime = percentage * maxDuration;
        
        jumpToTime(targetTime);
        updateTimelineProgress(percentage * 100);
    });

    // Function to jump all audio players to specific time
    function jumpToTime(time) {
        let successCount = 0;
        audioElements.forEach((audio, index) => {
            // Wait for audio to load if needed
            if (audio.readyState >= 1) { // HAVE_METADATA or higher
                if (time <= audio.duration) {
                    // Round to nearest 0.1 second for more accurate positioning
                    audio.currentTime = Math.round(time * 10) / 10;
                    successCount++;
                }
                // Pause all audio to sync them
                audio.pause();
            } else {
                // If metadata not loaded, try to load and then set time
                audio.addEventListener('loadedmetadata', function() {
                    if (time <= audio.duration) {
                        audio.currentTime = Math.round(time * 10) / 10;
                    }
                    audio.pause();
                }, { once: true });
                audio.load();
            }
        });
        
        // Update timeline progress
        const maxDuration = 93; // 1:33 = 93 seconds
        const percentage = (time / maxDuration) * 100;
        updateTimelineProgress(percentage);
        
        // Show user feedback
        showTimelineMessage(`⏱️ Jumped to ${time.toFixed(1)}s`);
        console.log(`Timeline: Jumped ${successCount}/${audioElements.length} audio players to ${time}s`);
    }

    // Function to update timeline progress bar
    function updateTimelineProgress(percentage) {
        if (timelineProgress) {
            timelineProgress.style.width = Math.min(percentage, 100) + '%';
        }
    }

    // Function to show user feedback message
    function showTimelineMessage(message) {
        // Create or update feedback message
        let feedbackElement = document.querySelector('.timeline-feedback');
        if (!feedbackElement) {
            feedbackElement = document.createElement('div');
            feedbackElement.className = 'timeline-feedback';
            feedbackElement.style.cssText = `
                position: absolute;
                top: -50px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--primary-color);
                color: white;
                padding: 0.6rem 1.2rem;
                border-radius: 8px;
                font-size: 0.875rem;
                font-weight: 500;
                opacity: 0;
                transition: all 0.3s ease;
                pointer-events: none;
                z-index: 20;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                white-space: nowrap;
            `;
            document.querySelector('.timeline-wrapper').appendChild(feedbackElement);
        }
        
        feedbackElement.textContent = message;
        feedbackElement.style.opacity = '1';
        feedbackElement.style.transform = 'translateX(-50%) translateY(-4px)';
        
        // Hide message after 2 seconds
        setTimeout(() => {
            feedbackElement.style.opacity = '0';
            feedbackElement.style.transform = 'translateX(-50%) translateY(0px)';
        }, 2000);
    }

    // Track audio progress and update timeline (optional feature)
    let activeAudio = null;
    
    audioElements.forEach(audio => {
        audio.addEventListener('play', function() {
            activeAudio = this;
        });
        
        audio.addEventListener('pause', function() {
            if (activeAudio === this) {
                activeAudio = null;
            }
        });
        
        audio.addEventListener('timeupdate', function() {
            if (this.currentTime > 0 && this.duration && activeAudio === this) {
                const percentage = (this.currentTime / this.duration) * 100;
                updateTimelineProgress(percentage);
            }
        });
        
        // Reset timeline when audio ends
        audio.addEventListener('ended', function() {
            if (activeAudio === this) {
                updateTimelineProgress(0);
                activeAudio = null;
            }
        });
    });
    
    console.log('Audio timeline initialized successfully');
}
