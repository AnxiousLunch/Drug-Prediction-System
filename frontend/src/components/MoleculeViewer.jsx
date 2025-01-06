import { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { GlitchPass } from 'three/examples/jsm/postprocessing/GlitchPass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default function MoleculeViewer() {
  const containerRef = useRef();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.001);

    const camera = new THREE.PerspectiveCamera(75, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.5;
    containerRef.current.appendChild(renderer.domElement);

    // Enhanced post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      2.0, 0.5, 0.9
    );
    composer.addPass(bloomPass);

    const glitchPass = new GlitchPass();
    glitchPass.goWild = false;
    glitchPass.curF = 0.5;
    composer.addPass(glitchPass);

    // Create enhanced DNA structure
    const dnaGroup = new THREE.Group();
    const helixRadius = 4;
    const helixHeight = 12;
    const helixPoints = 60;

    // Add base pairs with enhanced materials
    for (let i = 0; i < helixPoints; i++) {
      const t = i / helixPoints;
      const angle = t * Math.PI * 4;

      // Enhanced material for strands
      const strandMaterial = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color().setHSL(t, 1, 0.5),
        metalness: 0.8,
        roughness: 0.2,
        emissive: new THREE.Color().setHSL(t, 1, 0.3),
        emissiveIntensity: 0.5
      });

      // First strand with enhanced geometry
      const sphere1 = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.25, 2),
        strandMaterial
      );
      sphere1.position.set(
        Math.cos(angle) * helixRadius,
        t * helixHeight - helixHeight / 2,
        Math.sin(angle) * helixRadius
      );
      dnaGroup.add(sphere1);

      // Second strand
      const sphere2 = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.25, 2),
        new THREE.MeshPhysicalMaterial({
          color: new THREE.Color().setHSL(t + 0.5, 1, 0.5),
          metalness: 0.8,
          roughness: 0.2,
          emissive: new THREE.Color().setHSL(t + 0.5, 1, 0.3),
          emissiveIntensity: 0.5
        })
      );
      sphere2.position.set(
        Math.cos(angle + Math.PI) * helixRadius,
        t * helixHeight - helixHeight / 2,
        Math.sin(angle + Math.PI) * helixRadius
      );
      dnaGroup.add(sphere2);

      // Enhanced connections
      if (i < helixPoints - 1) {
        const connection = new THREE.Mesh(
          new THREE.CylinderGeometry(0.05, 0.05, helixRadius * 2, 8),
          new THREE.MeshPhysicalMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.4,
            metalness: 0.9,
            roughness: 0.1
          })
        );
        connection.position.set(0, t * helixHeight - helixHeight / 2, 0);
        connection.rotation.x = Math.PI / 2;
        connection.rotation.z = angle;
        dnaGroup.add(connection);
      }
    }

    scene.add(dnaGroup);

    // Enhanced particle system
    const particleCount = 3000;
    const particleGeometry = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);
    const particleSpeeds = new Float32Array(particleCount);
    const particleSizes = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      particlePositions[i3] = (Math.random() - 0.5) * 60;
      particlePositions[i3 + 1] = (Math.random() - 0.5) * 60;
      particlePositions[i3 + 2] = (Math.random() - 0.5) * 60;

      const color = new THREE.Color();
      color.setHSL(Math.random(), 1.0, 0.6);
      particleColors[i3] = color.r;
      particleColors[i3 + 1] = color.g;
      particleColors[i3 + 2] = color.b;

      particleSpeeds[i] = Math.random() * 2 + 1;
      particleSizes[i] = Math.random() * 0.2 + 0.1;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(particleSizes, 1));

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true
    });

    const particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particleSystem);

    // Enhanced lighting
    const lights = Array(8).fill().map(() => {
      const light = new THREE.PointLight(0xffffff, 1.5);
      scene.add(light);
      return light;
    });

    scene.add(new THREE.AmbientLight(0x404040, 2));

    camera.position.z = 18;

    // Orbit controls for better interaction
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.maxPolarAngle = Math.PI / 2;

    setTimeout(() => setIsLoading(false), 1000);

    let time = 0;
    function animate() {
      requestAnimationFrame(animate);
      time += 0.005;

      // Enhanced DNA animation
      dnaGroup.rotation.y += 0.003;
      dnaGroup.rotation.x = Math.sin(time * 0.3) * 0.15;
      dnaGroup.position.y = Math.sin(time) * 0.5;

      // Enhanced particle animation
      const positions = particleSystem.geometry.attributes.position.array;
      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        positions[i3 + 1] += particleSpeeds[i] * 0.015;
        if (positions[i3 + 1] > 30) positions[i3 + 1] = -30;

        positions[i3] += Math.sin(time + i) * 0.02;
        positions[i3 + 2] += Math.cos(time + i) * 0.02;
      }
      particleSystem.geometry.attributes.position.needsUpdate = true;

      // Enhanced light animation
      lights.forEach((light, i) => {
        const radius = 25;
        light.position.x = Math.cos(time * 0.4 + i) * radius;
        light.position.y = Math.sin(time * 0.2 + i) * radius;
        light.position.z = Math.sin(time * 0.6 + i) * radius;
        light.intensity = 1.5 + Math.sin(time * 1.5 + i) * 0.5;
      });

      controls.update(); // Update controls
      composer.render();
    }
    animate();

    function handleResize() {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      composer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    }
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      containerRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
      composer.dispose();
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-90">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500" />
        </div>
      )}
    </div>
  );
}