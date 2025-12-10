import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// Init Scene
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
// Add some fog for depth
scene.fog = new THREE.FogExp2(0x1a1a1a, 0.02);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(20, 20, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// Lights
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 3);
hemiLight.position.set(0, 20, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 3);
dirLight.position.set(3, 10, 10);
scene.add(dirLight);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 5, 0); // Focus around where our meshes are

// Objects
let mandibleMesh;
let maxillaMesh;

// Loader
const loader = new STLLoader();

function log(msg) {
    const el = document.getElementById('log');
    el.innerHTML += `<div>${msg}</div>`;
    el.scrollTop = el.scrollHeight;
}

// Load Meshes
// Note: We access assets via /assets/
loader.load('/assets/dummy_maxilla.stl', function (geometry) {
    const material = new THREE.MeshPhongMaterial({ color: 0xaaaaaa, opacity: 0.5, transparent: true, side: THREE.DoubleSide });
    maxillaMesh = new THREE.Mesh(geometry, material);
    scene.add(maxillaMesh);
    log("Loaded Maxilla");
}, undefined, function (error) {
    console.error(error);
    log("Error loading Maxilla");
});

loader.load('/assets/dummy_mandible.stl', function (geometry) {
    const material = new THREE.MeshPhongMaterial({ color: 0x4facfe, specular: 0x111111, shininess: 200 });
    mandibleMesh = new THREE.Mesh(geometry, material);

    // IMPORTANT: We will control the matrix manually
    mandibleMesh.matrixAutoUpdate = false;

    scene.add(mandibleMesh);
    log("Loaded Mandible");
}, undefined, function (error) {
    console.error(error);
    log("Error loading Mandible");
});

// Sync State Loop
async function syncState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Update Metrics
        if (data.reward !== undefined) {
            document.getElementById('val-reward').innerText = data.reward.toFixed(3);
        }
        if (data.metrics) {
            const m = data.metrics;

            // Helper to safe update
            const update = (id, val) => {
                const el = document.getElementById(id);
                if (el && val !== undefined) el.innerText = (typeof val === 'number') ? val.toFixed(2) : val;
            };

            update('val-overjet', m.overjet_mm);
            update('val-overbite', m.overbite_mm);
            update('val-midline', m.midline_dev_mm);
            update('val-ant-contact', m.anterior_contact_ratio);
            update('val-post-contact', m.posterior_contact_ratio);
            update('val-openbite', m.anterior_openbite_fraction);

            // Can add more if needed
        }

        // Update Transform
        if (data.matrix && mandibleMesh) {
            // data.matrix is 4x4 array (row-major or column-major?)
            // Trimesh/NumPy likely row-major. Three.js is column-major.
            // We might need to transpose if coming from numpy

            // Checking: Trimesh transform matrices are standard homogenous matrices.
            // Three.js matrix.set() takes row-major arguments n11, n12 ... 
            // set( n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 )

            // Assume data.matrix is nested list [[r,r,r,t],[...]]
            const m = data.matrix;

            // Flatten if needed or access directly
            // matrix.set takes elements in row-major order.

            mandibleMesh.matrix.set(
                m[0][0], m[0][1], m[0][2], m[0][3],
                m[1][0], m[1][1], m[1][2], m[1][3],
                m[2][0], m[2][1], m[2][2], m[2][3],
                m[3][0], m[3][1], m[3][2], m[3][3]
            );
        }
    } catch (e) {
        console.warn("Sync error", e);
    }
}

// Polling
setInterval(syncState, 100);

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

// Resize Handler
window.addEventListener('resize', onWindowResize, false);
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Window Functions for UI
window.resetSim = async () => {
    log("Resetting...");
    await fetch('/api/reset', { method: 'POST' });
};

window.runICP = async () => {
    log("Running ICP...");
    await fetch('/api/icp', { method: 'POST' });
    log("ICP Done.");
};

window.stepRandom = async () => {
    // Random tiny move
    const act = {
        dRx: (Math.random() - 0.5) * 2, // +/- 1 deg
        dRy: (Math.random() - 0.5) * 2,
        dRz: (Math.random() - 0.5) * 2,
        dTx: (Math.random() - 0.5) * 0.5, // +/- 0.25 mm
        dTy: (Math.random() - 0.5) * 0.5,
        dTz: (Math.random() - 0.5) * 0.5
    };

    await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(act)
    });
};

window.runAuto = async () => {
    log("Running Auto...");
    for (let i = 0; i < 100; i++) {
        await window.stepRandom();
        // wait a bit
        await new Promise(r => setTimeout(r, 50));
    }
    log("Done.");
};

window.runRL = async () => {
    log("Running RL Fine-tuning...");

    const MAX_STEPS = 2000;
    const CONVERGENCE_THRESHOLD = 0.0001; // Stop if reward improvement < this
    let prevReward = -Infinity;

    for (let i = 0; i < MAX_STEPS; i++) {
        try {
            const res = await fetch('/api/rl_step', { method: 'POST' });
            const data = await res.json();

            if (data.error) {
                log("Error: " + data.error);
                break;
            }

            // Convergence Check REMOVED per user request
            // const currentReward = data.reward;
            // if (Math.abs(currentReward - prevReward) < CONVERGENCE_THRESHOLD) { ... }
            // prevReward = currentReward;


            // Wait slightly for animation effect
            await new Promise(r => setTimeout(r, 50));

        } catch (e) {
            console.error(e);
            break;
        }
    }
    log("RL Done.");
};
