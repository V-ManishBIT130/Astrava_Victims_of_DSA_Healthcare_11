import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { ArrowUp, Shield, Lock, Eye, Users, ArrowRight, Mic, Volume2, VolumeX, MicOff, Globe } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import translations, { LANG_TO_BCP47 } from './translations';

/* ═══════════════════════════════════════════════════════
   COLOR TOKENS — Blue palette, light background
   ═══════════════════════════════════════════════════════ */
const C = {
  bg:         '#f8f9fc',
  surface:    '#ffffff',
  surface2:   '#eef1f8',
  border:     '#d6dce8',
  borderLight:'#e5e9f2',
  heading:    '#0f172a',
  body:       '#334155',
  muted:      '#94a3b8',
  accent:     '#2563eb',
  accentDark: '#1d4ed8',
  accentLight:'#3b82f6',
  accentBg:   '#eff4ff',
  accentMuted:'#dbeafe',
  chatBotBg:  '#f1f5f9',
  chatUserBg: '#2563eb',
  chatHeader: '#0f172a',
};

const font = {
  serif: "'Instrument Serif', serif",
  mono:  "'JetBrains Mono', monospace",
  body:  "'DM Sans', sans-serif",
};

/* ═══════════════════════════════════════════════════════
   ANIMATION WRAPPER — stagger-in on scroll
   ═══════════════════════════════════════════════════════ */
function Reveal({ children, className, delay = 0 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 32 }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   TYPING INDICATOR
   ═══════════════════════════════════════════════════════ */
function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="flex items-end gap-3 mb-4"
    >
      <div
        className="flex items-center gap-1.5 px-5 py-4 rounded-2xl rounded-bl-sm"
        style={{ background: C.chatBotBg }}
      >
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="rounded-full"
            style={{ width: 6, height: 6, background: C.muted }}
            animate={{ y: [0, -5, 0], opacity: [0.4, 1, 0.4] }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: i * 0.18,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   MESSAGE BUBBLE
   ═══════════════════════════════════════════════════════ */
function MessageBubble({ message, isBot, isSpeaking, onSpeak, t }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className={`flex mb-3 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      <div
        className="max-w-[82%] sm:max-w-[70%] px-5 py-3.5"
        style={{
          background: isBot ? C.chatBotBg : C.chatUserBg,
          borderRadius: isBot ? '20px 20px 20px 6px' : '20px 20px 6px 20px',
          color: isBot ? C.body : '#ffffff',
          fontFamily: font.body,
          fontSize: 15,
          fontWeight: 400,
          lineHeight: 1.65,
        }}
      >
        {message}
        {isBot && onSpeak && (
          <button
            onClick={onSpeak}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              padding: '2px 6px',
              marginLeft: 8,
              opacity: 0.5,
              verticalAlign: 'middle',
              transition: 'opacity 0.2s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.5'; }}
            title={isSpeaking ? (t ? t('speakStop') : 'Stop speaking') : (t ? t('speakStart') : 'Read aloud')}
          >
            {isSpeaking
              ? <VolumeX size={14} style={{ color: C.accent }} />
              : <Volume2 size={14} style={{ color: C.muted }} />
            }
          </button>
        )}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   QUICK REPLY CHIPS
   ═══════════════════════════════════════════════════════ */
function QuickChips({ onSelect, visible, t }) {
  const chips = [
    t('chip1'),
    t('chip2'),
    t('chip3'),
    t('chip4'),
  ];
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-wrap gap-2 mb-4 ml-1"
        >
          {chips.map((chip, i) => (
            <motion.button
              key={chip}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 + i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              onClick={() => onSelect(chip)}
              className="cursor-pointer"
              style={{
                fontFamily: font.body,
                fontSize: 14,
                fontWeight: 400,
                color: C.accent,
                background: C.accentBg,
                border: `1px solid ${C.accentMuted}`,
                borderRadius: 24,
                padding: '9px 18px',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = C.accentMuted;
                e.currentTarget.style.borderColor = C.accentLight;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = C.accentBg;
                e.currentTarget.style.borderColor = C.accentMuted;
              }}
            >
              {chip}
            </motion.button>
          ))}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ────────────────────────────────────────────────────────
   FEATURES CAROUSEL - infinite auto-scroll marquee
   ──────────────────────────────────────────────────────── */
function FeaturesCarousel({ features, bgColor }) {
  const [paused, setPaused] = useState(false);
  const doubled = [...features, ...features];

  return (
    <div
      style={{ overflow: 'hidden', position: 'relative', padding: '8px 0' }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      {/* Left fade edge */}
      <div
        style={{
          position: 'absolute', left: 0, top: 0, bottom: 0, width: 100,
          background: `linear-gradient(to right, ${bgColor ?? C.bg}, transparent)`,
          zIndex: 2, pointerEvents: 'none',
        }}
      />
      {/* Right fade edge */}
      <div
        style={{
          position: 'absolute', right: 0, top: 0, bottom: 0, width: 100,
          background: `linear-gradient(to left, ${bgColor ?? C.bg}, transparent)`,
          zIndex: 2, pointerEvents: 'none',
        }}
      />

      <div
        className={`marquee-track${paused ? ' paused' : ''}`}
        style={{ display: 'flex', gap: 24, width: 'max-content', alignItems: 'stretch' }}
      >
        {doubled.map((f, i) => (
          <div
            key={i}
            className="feature-card"
            style={{
              width: 300,
              padding: '32px',
              borderRadius: 16,
              background: C.surface,
              border: `1px solid ${C.borderLight}`,
              flexShrink: 0,
              cursor: 'default',
            }}
          >
            <div
              style={{
                width: 48, height: 48, borderRadius: 12,
                background: C.accentBg,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: 20,
              }}
            >
              <f.icon style={{ width: 20, height: 20, color: C.accent }} />
            </div>
            <h3
              style={{
                fontFamily: font.body,
                fontSize: 19, fontWeight: 600,
                color: C.heading, marginBottom: 8, margin: '0 0 8px',
              }}
            >
              {f.title}
            </h3>
            <p
              style={{
                fontFamily: font.body,
                fontSize: 15, lineHeight: 1.65,
                color: C.body, margin: 0,
              }}
            >
              {f.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────
   LANDING PAGE
   ──────────────────────────────────────────────────────── */
function LandingPage({ onStart, t, appLang, setAppLang }) {
  const features = [
    {
      icon: Lock,
      title: t('feat1Title'),
      desc: t('feat1Desc'),
    },
    {
      icon: Shield,
      title: t('feat2Title'),
      desc: t('feat2Desc'),
    },
    {
      icon: Eye,
      title: t('feat3Title'),
      desc: t('feat3Desc'),
    },
    {
      icon: Users,
      title: t('feat4Title'),
      desc: t('feat4Desc'),
    },
  ];

  return (
    <motion.div
      key="landing"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen"
      style={{ background: C.bg }}
    >
      {/* ─── NAV ─── */}
      <nav
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 sm:px-10"
        style={{
          height: 64,
          background: 'rgba(248,249,252,0.85)',
          backdropFilter: 'blur(12px)',
          borderBottom: `1px solid ${C.borderLight}`,
        }}
      >
        <span
          style={{
            fontFamily: font.serif,
            fontSize: 24,
            fontWeight: 400,
            color: C.heading,
          }}
        >
          {t('brand')}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <LanguageSelector appLang={appLang} setAppLang={setAppLang} />
          <Button
            onClick={onStart}
            className="cursor-pointer"
            style={{
              fontFamily: font.body,
              fontSize: 14,
              fontWeight: 500,
              background: C.accent,
              color: '#fff',
              borderRadius: 8,
              padding: '10px 24px',
            }}
          >
            {t('navCta')}
          </Button>
        </div>
      </nav>

      {/* ─── HERO ─── */}
      <section
        className="flex flex-col items-center justify-center text-center px-6"
        style={{ minHeight: '100vh', paddingTop: 64 }}
      >
        {/* Decorative animated gradient circle behind heading */}
        <motion.div
          className="absolute rounded-full pointer-events-none"
          style={{
            width: 500,
            height: 500,
            background: `radial-gradient(circle, ${C.accentMuted} 0%, transparent 70%)`,
            filter: 'blur(60px)',
            opacity: 0.5,
          }}
          animate={{
            scale: [1, 1.12, 1],
            opacity: [0.4, 0.6, 0.4],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        />

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          className="relative z-10"
        >
          <h1
            style={{
              fontFamily: font.serif,
              fontSize: 'clamp(40px, 7vw, 72px)',
              fontWeight: 400,
              lineHeight: 1.1,
              letterSpacing: '-0.02em',
              color: C.heading,
              margin: 0,
              marginBottom: 24,
            }}
          >
            {t('heroHeading1')}
            <br />
            <span style={{ fontStyle: 'italic', color: C.accent }}>{t('heroHeading2')}</span>
          </h1>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35, ease: [0.22, 1, 0.36, 1] }}
          className="relative z-10"
          style={{
            fontFamily: font.body,
            fontSize: 'clamp(17px, 2.2vw, 20px)',
            fontWeight: 400,
            lineHeight: 1.65,
            color: C.body,
            maxWidth: 520,
            margin: '0 auto',
            marginBottom: 48,
          }}
        >
          {t('heroSub')}
        </motion.p>

        <motion.div
          initial={{ opacity: 0, scale: 0.92 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="relative z-10"
        >
          <motion.button
            onClick={onStart}
            className="cursor-pointer group"
            style={{
              fontFamily: font.body,
              fontSize: 20,
              fontWeight: 600,
              color: '#ffffff',
              background: C.accent,
              border: 'none',
              borderRadius: 16,
              padding: '22px 56px',
              boxShadow: '0 4px 24px rgba(37,99,235,0.3), 0 1px 3px rgba(37,99,235,0.2)',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
            whileHover={{ scale: 1.04, boxShadow: '0 8px 36px rgba(37,99,235,0.4)' }}
            whileTap={{ scale: 0.97 }}
          >
            {t('heroCta')}
            <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
          </motion.button>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 1 }}
          className="relative z-10"
          style={{
            fontFamily: font.mono,
            fontSize: 12,
            letterSpacing: '0.06em',
            color: C.muted,
            marginTop: 20,
          }}
        >
          {t('heroNote')}
        </motion.p>

        {/* Scroll arrow */}
        <motion.div
          className="absolute bottom-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1, y: [0, 8, 0] }}
          transition={{
            opacity: { delay: 1.5, duration: 0.6 },
            y: { delay: 1.5, duration: 2.5, repeat: Infinity, ease: 'easeInOut' },
          }}
        >
          <svg width="20" height="32" viewBox="0 0 20 32" fill="none">
            <rect x="1" y="1" width="18" height="30" rx="9" stroke={C.muted} strokeWidth="1.5" />
            <motion.circle
              cx="10" cy={8} r="3" fill={C.accent}
              initial={{ cy: 8 }}
              animate={{ cy: [8, 18, 8] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            />
          </svg>
        </motion.div>
      </section>

      {/* ─── AFFIRMING SECTION ─── */}
      <section className="py-20 sm:py-28 px-6" style={{ background: C.surface }}>
        <div className="max-w-3xl mx-auto text-center">
          <Reveal>
            <h2
              style={{
                fontFamily: font.serif,
                fontSize: 'clamp(28px, 4.5vw, 42px)',
                fontWeight: 400,
                lineHeight: 1.15,
                color: C.heading,
                marginBottom: 20,
              }}
            >
              {t('affirmHeading1')}
              <span style={{ fontStyle: 'italic', color: C.accent }}>{t('affirmHighlight')}</span>
              {t('affirmHeading2')}
            </h2>
          </Reveal>
          <Reveal delay={0.15}>
            <p
              style={{
                fontFamily: font.body,
                fontSize: 18,
                fontWeight: 400,
                lineHeight: 1.7,
                color: C.body,
                maxWidth: 560,
                margin: '0 auto',
              }}
            >
              {t('affirmBody')}
            </p>
          </Reveal>
        </div>
      </section>

      {/* Features - Trust & Safety */}
      <section className="py-20 sm:py-28 px-6" style={{ background: C.bg }}>
        <div className="max-w-5xl mx-auto">
          <Reveal>
            <div className="text-center mb-16">
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: 12,
                  letterSpacing: '0.12em',
                  color: C.accent,
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: 12,
                }}
              >
                {t('featuresBadge')}
              </span>
              <h2
                style={{
                  fontFamily: font.serif,
                  fontSize: 'clamp(28px, 4.5vw, 40px)',
                  fontWeight: 400,
                  color: C.heading,
                }}
              >
                {t('featuresHeading1')}
                <span style={{ fontStyle: 'italic', color: C.accent }}>{t('featuresHighlight')}</span>
              </h2>
            </div>
          </Reveal>

          <FeaturesCarousel features={features} bgColor={C.bg} />
        </div>
      </section>

      {/* ─── HOW IT WORKS ─── */}
      <section className="py-20 sm:py-28 px-6" style={{ background: C.surface }}>
        <div className="max-w-3xl mx-auto">
          <Reveal>
            <div className="text-center mb-16">
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: 12,
                  letterSpacing: '0.12em',
                  color: C.accent,
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: 12,
                }}
              >
                {t('stepsBadge')}
              </span>
              <h2
                style={{
                  fontFamily: font.serif,
                  fontSize: 'clamp(28px, 4.5vw, 40px)',
                  fontWeight: 400,
                  color: C.heading,
                }}
              >
                {t('stepsHeading')}
              </h2>
            </div>
          </Reveal>

          {[
            { num: '01', title: t('step1Title'), desc: t('step1Desc') },
            { num: '02', title: t('step2Title'), desc: t('step2Desc') },
            { num: '03', title: t('step3Title'), desc: t('step3Desc') },
          ].map((step, i) => (
            <Reveal key={i} delay={i * 0.12}>
              <div
                className="flex gap-6 sm:gap-8 items-start py-8"
                style={{
                  borderBottom: i < 2 ? `1px solid ${C.borderLight}` : 'none',
                }}
              >
                <span
                  style={{
                    fontFamily: font.serif,
                    fontSize: 48,
                    fontWeight: 400,
                    color: C.accentMuted,
                    lineHeight: 1,
                    flexShrink: 0,
                    minWidth: 56,
                  }}
                >
                  {step.num}
                </span>
                <div>
                  <h3
                    style={{
                      fontFamily: font.body,
                      fontSize: 19,
                      fontWeight: 600,
                      color: C.heading,
                      marginBottom: 6,
                    }}
                  >
                    {step.title}
                  </h3>
                  <p
                    style={{
                      fontFamily: font.body,
                      fontSize: 16,
                      fontWeight: 400,
                      lineHeight: 1.65,
                      color: C.body,
                      margin: 0,
                    }}
                  >
                    {step.desc}
                  </p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ─── BOTTOM CTA ─── */}
      <section className="py-24 sm:py-32 px-6" style={{ background: C.bg }}>
        <div className="max-w-2xl mx-auto text-center">
          <Reveal>
            <h2
              style={{
                fontFamily: font.serif,
                fontSize: 'clamp(30px, 5vw, 48px)',
                fontWeight: 400,
                lineHeight: 1.15,
                color: C.heading,
                marginBottom: 16,
              }}
            >
              {t('ctaHeading1')}
              <br />
              {t('ctaHeading2')}
              <span style={{ fontStyle: 'italic', color: C.accent }}>{t('ctaHighlight')}</span>
              {t('ctaHeading3')}
            </h2>
          </Reveal>
          <Reveal delay={0.15}>
            <p
              style={{
                fontFamily: font.body,
                fontSize: 18,
                fontWeight: 400,
                lineHeight: 1.7,
                color: C.body,
                maxWidth: 460,
                margin: '0 auto 40px',
              }}
            >
              {t('ctaBody')}
            </p>
          </Reveal>
          <Reveal delay={0.3}>
            <motion.button
              onClick={onStart}
              className="cursor-pointer group"
              style={{
                fontFamily: font.body,
                fontSize: 20,
                fontWeight: 600,
                color: '#ffffff',
                background: C.accent,
                border: 'none',
                borderRadius: 16,
                padding: '22px 56px',
                boxShadow: '0 4px 24px rgba(37,99,235,0.3)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 12,
              }}
              whileHover={{ scale: 1.04, boxShadow: '0 8px 36px rgba(37,99,235,0.4)' }}
              whileTap={{ scale: 0.97 }}
            >
              {t('ctaBtn')}
              <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
            </motion.button>
          </Reveal>
        </div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer
        className="py-8 px-6 text-center"
        style={{
          borderTop: `1px solid ${C.borderLight}`,
          background: C.surface,
        }}
      >
        <p
          style={{
            fontFamily: font.mono,
            fontSize: 11,
            letterSpacing: '0.06em',
            color: C.muted,
          }}
        >
          {t('footer')}
        </p>
      </footer>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   CHAT PAGE
   ═══════════════════════════════════════════════════════ */
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

// stable session id for this browser tab
const SESSION_ID = (() => {
  let id = sessionStorage.getItem('astrava_session');
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem('astrava_session', id); }
  return id;
})();

/* ════════════════════════════════════════════════════════
   THERAPIST CARD — 3-step flow for MEDIUM risk
   ════════════════════════════════════════════════════════ */
function TherapistCard({ onClose, onDone, t }) {
  // 'ask' → 'share' → 'notified' | 'notified_no_data'
  const [step, setStep] = useState('ask');
  const [submitting, setSubmitting] = useState(false);

  const notifyTherapist = async (shareData) => {
    setSubmitting(true);
    try {
      await fetch(`${API_URL}/api/request-therapist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:   SESSION_ID,
          migrate_chat: shareData,
        }),
      });
    } catch (_) {}
    setStep(shareData ? 'notified' : 'notified_no_data');
    setSubmitting(false);
    setTimeout(onDone, 4000);
  };

  const primaryBtn = {
    fontFamily: font.body, fontSize: 13, fontWeight: 600,
    background: C.accent, color: '#fff', border: 'none',
    borderRadius: 8, padding: '9px 18px', cursor: 'pointer',
  };
  const ghostBtn = {
    fontFamily: font.body, fontSize: 13, fontWeight: 400,
    background: 'transparent', color: C.muted,
    border: `1px solid ${C.border}`, borderRadius: 8,
    padding: '9px 16px', cursor: 'pointer',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: C.surface,
        border: `1px solid ${C.borderLight}`,
        borderRadius: 16,
        padding: '20px 24px',
        marginBottom: 20,
        fontFamily: font.body,
      }}
    >
      {step === 'ask' && (
        <>
          <p style={{ fontSize: 14, color: C.body, lineHeight: 1.65, margin: '0 0 16px' }}>
            {t('therapistAsk')}
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <button onClick={() => setStep('share')} style={primaryBtn}>{t('therapistYes')}</button>
            <button onClick={onClose} style={ghostBtn}>{t('therapistNo')}</button>
          </div>
        </>
      )}

      {step === 'share' && (
        <>
          <p style={{ fontSize: 14, color: C.body, lineHeight: 1.65, margin: '0 0 16px' }}>
            {t('therapistShare')}
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => notifyTherapist(true)}
              disabled={submitting}
              style={{ ...primaryBtn, opacity: submitting ? 0.7 : 1 }}
            >
              {submitting ? t('therapistNotifying') : t('therapistYes')}
            </button>
            <button
              onClick={() => notifyTherapist(false)}
              disabled={submitting}
              style={{ ...ghostBtn, opacity: submitting ? 0.7 : 1 }}
            >
              {t('therapistNo')}
            </button>
          </div>
        </>
      )}

      {step === 'notified' && (
        <p style={{ fontSize: 14, color: C.body, lineHeight: 1.65, margin: 0 }}>
          {t('therapistNotified')}
        </p>
      )}

      {step === 'notified_no_data' && (
        <p style={{ fontSize: 14, color: C.body, lineHeight: 1.65, margin: 0 }}>
          {t('therapistNotifiedNoData')}
        </p>
      )}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   CRISIS OVERLAY — blocks chat and shows help
   ═══════════════════════════════════════════════════════ */
function CrisisOverlay({ hasLocation, onResume, t }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'rgba(15,23,42,0.85)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 20,
      }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.92, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
        style={{
          background: '#ffffff',
          borderRadius: 20,
          maxWidth: 460,
          width: '100%',
          padding: '36px 32px 28px',
          boxShadow: '0 32px 64px rgba(0,0,0,0.25)',
          textAlign: 'center',
        }}
      >
        {/* Red alert icon */}
        <div style={{
          width: 56, height: 56, borderRadius: '50%',
          background: '#fef2f2', display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 20px',
        }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>

        <h2 style={{
          fontFamily: font.serif, fontSize: 26, fontWeight: 400,
          color: '#0f172a', margin: '0 0 10px',
        }}>
          {t('crisisTitle')}
        </h2>

        <p style={{
          fontFamily: font.body, fontSize: 15, color: '#475569',
          lineHeight: 1.65, margin: '0 0 20px',
        }}>
          {t('crisisBody')}
        </p>

        {hasLocation && (
          <div style={{
            background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 10,
            padding: '10px 16px', marginBottom: 16,
            fontFamily: font.body, fontSize: 13, color: '#15803d',
            display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            {t('crisisLocationShared')}
          </div>
        )}

        {/* Helplines */}
        <div style={{
          background: '#f8fafc', borderRadius: 12, padding: '16px 20px',
          marginBottom: 20, textAlign: 'left',
        }}>
          <p style={{
            fontFamily: font.mono, fontSize: 11, letterSpacing: '0.06em',
            color: '#94a3b8', textTransform: 'uppercase', margin: '0 0 10px',
          }}>
            {t('crisisHelplines')}
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { name: 'iCall (India)', number: '9152987821' },
              { name: 'Vandrevala Foundation', number: '18602662345' },
              { name: 'AASRA', number: '9820466726' },
            ].map((h) => (
              <a
                key={h.number}
                href={`tel:${h.number}`}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  fontFamily: font.body, fontSize: 14, color: '#0f172a',
                  textDecoration: 'none', padding: '8px 12px',
                  background: '#ffffff', borderRadius: 8, border: '1px solid #e2e8f0',
                }}
              >
                <span>{h.name}</span>
                <span style={{ fontFamily: font.mono, fontSize: 13, color: '#2563eb', fontWeight: 600 }}>
                  {h.number}
                </span>
              </a>
            ))}
          </div>
        </div>

        <p style={{
          fontFamily: font.body, fontSize: 13, color: '#64748b',
          lineHeight: 1.6, margin: '0 0 20px',
        }}>
          {t('crisisHelp')}
        </p>

        <button
          onClick={onResume}
          style={{
            fontFamily: font.body, fontSize: 13, fontWeight: 400,
            background: 'transparent', color: '#94a3b8',
            border: `1px solid #e2e8f0`, borderRadius: 8,
            padding: '9px 20px', cursor: 'pointer',
          }}
        >
          {t('crisisResume')}
        </button>
      </motion.div>
    </motion.div>
  );
}

function ChatPage({ t, appLang, setAppLang }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showChips, setShowChips] = useState(false);
  const [userMsgCount, setUserMsgCount] = useState(0);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showSaveBtn, setShowSaveBtn] = useState(false);
  const [startTime] = useState(Date.now());
  // location & alert
  const [userLocation, setUserLocation] = useState(null);  // { lat, lng }
  const [locationBanner, setLocationBanner] = useState('idle'); // 'idle' | 'asking' | 'granted' | 'denied'
  const [alertSent, setAlertSent] = useState(false);
  // crisis overlay
  const [crisisActive, setCrisisActive] = useState(false);
  // therapist offer
  const [showTherapistCard, setShowTherapistCard] = useState(false);
  const [therapistDone, setTherapistDone] = useState(false);
  // voice
  const [isListening, setIsListening] = useState(false);
  const [detectedLang, setDetectedLang] = useState(null);     // BCP-47 from STT
  const [speakingMsgId, setSpeakingMsgId] = useState(null);    // which bot msg is being spoken
  const recognitionRef = useRef(null);
  const voicesRef = useRef([]);  // preloaded TTS voices
  const fallbackAudioRef = useRef(null); // for Google Translate TTS fallback
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const hasSaveTriggered = useRef(false);

  // Preload TTS voices (they load async in most browsers)
  useEffect(() => {
    const loadVoices = () => {
      const v = window.speechSynthesis.getVoices();
      if (v.length) voicesRef.current = v;
    };
    loadVoices();
    window.speechSynthesis.addEventListener('voiceschanged', loadVoices);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', loadVoices);
  }, []);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      const el = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, []);

  // Request geolocation — shown as a gentle banner, never forced
  const requestLocation = useCallback(() => {
    if (!navigator.geolocation) { setLocationBanner('denied'); return; }
    setLocationBanner('asking');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setLocationBanner('granted');
      },
      () => setLocationBanner('denied'),
      { timeout: 10000 },
    );
  }, []);

  // Offer location banner 3 s after chat loads (non-intrusive timing)
  useEffect(() => {
    const t = setTimeout(() => {
      if (locationBanner === 'idle') setLocationBanner('prompt');
    }, 3000);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setTimeout(scrollToBottom, 50);
  }, [messages, isTyping, scrollToBottom]);

  const now = () => {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Opening sequence
  useEffect(() => {
    const t1 = setTimeout(() => setIsTyping(true), 500);
    const t2 = setTimeout(() => {
      setIsTyping(false);
      setMessages([{
        id: 1,
        text: t('openingMsg1'),
        isBot: true,
        time: now(),
      }]);
      setShowChips(true);
    }, 500 + 1200);
    const t3 = setTimeout(() => setIsTyping(true), 500 + 1200 + 600);
    const t4 = setTimeout(() => {
      setIsTyping(false);
      setMessages((prev) => [...prev, {
        id: 2,
        text: t('openingMsg2'),
        isBot: true,
        time: now(),
      }]);
    }, 500 + 1200 + 600 + 1400);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); };
  }, []);

  const handleSend = useCallback(
    async (text) => {
      const trimmed = (text || input).trim();
      if (!trimmed || isTyping || crisisActive) return;

      const userMsg = { id: Date.now(), text: trimmed, isBot: false, time: now() };
      setMessages((prev) => [...prev, userMsg]);
      setInput('');
      setShowChips(false);
      if (textareaRef.current) textareaRef.current.style.height = 'auto';

      const newCount = userMsgCount + 1;
      setUserMsgCount(newCount);
      if (newCount >= 5) setShowSaveBtn(true);

      setIsTyping(true);
      try {
        const res = await fetch(`${API_URL}/api/chat`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            session_id: SESSION_ID,
            message:    trimmed,
            ...(userLocation && { location: userLocation }),
            language: LANG_TO_BCP47[appLang],
          }),
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        setIsTyping(false);

        if (data.alert_sent) setAlertSent(true);
        if (data.danger) setCrisisActive(true);
        if (data.ask_therapist_contact && !therapistDone) setShowTherapistCard(true);

        setMessages((prev) => [...prev, {
          id:       Date.now() + 1,
          text:     data.response,
          isBot:    true,
          time:     now(),
          // expose to UI if needed later
          criticality: data.criticality_label,
          danger:      data.danger,
          rag:         data.rag,
          inWarmup:    data.in_warmup,
        }]);

        // save-chat nudge after 5 messages (once)
        if (newCount >= 5 && !hasSaveTriggered.current) {
          hasSaveTriggered.current = true;
          setTimeout(() => setShowSaveDialog(true), 1200);
        }
      } catch (err) {
        setIsTyping(false);
        setMessages((prev) => [...prev, {
          id:    Date.now() + 1,
          text:  t('errorMsg'),
          isBot: true,
          time:  now(),
        }]);
      }
    },
    [input, isTyping, userMsgCount, crisisActive, appLang]
  );

  /* ── Voice input (Web Speech API — STT) ────────────────────────── */

  // Simple script-based language detection from transcript text
  const detectLanguage = (text) => {
    if (/[\u0900-\u097F]/.test(text)) return 'hi-IN';    // Devanagari → Hindi
    if (/[\u0C00-\u0C7F]/.test(text)) return 'te-IN';    // Telugu
    if (/[\u0C80-\u0CFF]/.test(text)) return 'kn-IN';    // Kannada
    if (/[\u0B80-\u0BFF]/.test(text)) return 'ta-IN';    // Tamil
    if (/[\u0D00-\u0D7F]/.test(text)) return 'ml-IN';    // Malayalam
    if (/[\u0980-\u09FF]/.test(text)) return 'bn-IN';    // Bengali
    if (/[\u0A00-\u0A7F]/.test(text)) return 'pa-IN';    // Punjabi
    if (/[\u0A80-\u0AFF]/.test(text)) return 'gu-IN';    // Gujarati
    if (/[\u0B00-\u0B7F]/.test(text)) return 'or-IN';    // Odia
    if (/[\u0600-\u06FF]/.test(text)) return 'ur-IN';    // Urdu/Arabic
    if (/[\u3040-\u30FF\u4E00-\u9FFF]/.test(text)) return 'ja-JP'; // Japanese
    if (/[\uAC00-\uD7AF]/.test(text)) return 'ko-KR';    // Korean
    if (/[\u0E00-\u0E7F]/.test(text)) return 'th-TH';    // Thai
    if (/[А-Яа-яЁё]/.test(text)) return 'ru-RU';         // Russian
    if (/[àâéèêëïîôùûüÿçœæ]/i.test(text)) return 'fr-FR'; // French
    if (/[äöüßÄÖÜ]/.test(text)) return 'de-DE';           // German
    if (/[ñáéíóúü¿¡]/i.test(text)) return 'es-ES';        // Spanish
    return 'en-US';
  };

  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error('[Voice] SpeechRecognition not supported in this browser');
      return;
    }

    const rec = new SpeechRecognition();
    rec.lang = LANG_TO_BCP47[appLang] || 'en-US';
    rec.continuous = false;
    rec.interimResults = false;
    rec.maxAlternatives = 1;

    rec.onstart = () => console.log('[Voice] Listening started, lang =', rec.lang);

    rec.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      console.log('[Voice] Got transcript:', transcript);
      const lang = detectLanguage(transcript);
      console.log('[Voice] Detected language:', lang);
      setDetectedLang(lang);
      setInput((prev) => prev ? prev + ' ' + transcript : transcript);
      setIsListening(false);
    };

    rec.onnomatch = () => {
      console.warn('[Voice] No speech match');
      setIsListening(false);
    };

    rec.onerror = (e) => {
      console.error('[Voice] STT error:', e.error, e.message);
      setIsListening(false);
    };

    rec.onend = () => {
      console.log('[Voice] Recognition ended');
      setIsListening(false);
    };

    recognitionRef.current = rec;
    try {
      rec.start();
      setIsListening(true);
    } catch (err) {
      console.error('[Voice] Failed to start:', err);
      setIsListening(false);
    }
  }, [isListening, appLang]);

  /* ── Voice output (Web Speech API — TTS) ────────────────────────── */

  // Server-side TTS proxy fallback for languages without a native browser voice
  const speakWithGoogleTTS = useCallback((text, langCode, msgId) => {
    // Backend proxy has 200 char limit; chunk the text
    const maxLen = 180;
    const chunks = [];
    let remaining = text;
    while (remaining.length > 0) {
      chunks.push(remaining.substring(0, maxLen));
      remaining = remaining.substring(maxLen);
    }

    setSpeakingMsgId(msgId);
    let idx = 0;

    const playNext = () => {
      if (idx >= chunks.length) {
        setSpeakingMsgId(null);
        fallbackAudioRef.current = null;
        return;
      }
      const encoded = encodeURIComponent(chunks[idx]);
      const url = `${API_URL}/api/tts?text=${encoded}&lang=${langCode}`;
      const audio = new Audio(url);
      fallbackAudioRef.current = audio;
      audio.onended = () => { idx++; playNext(); };
      audio.onerror = () => {
        console.error('[TTS Fallback] Audio error for chunk', idx);
        setSpeakingMsgId(null);
        fallbackAudioRef.current = null;
      };
      audio.play().catch((err) => {
        console.error('[TTS Fallback] Play failed:', err);
        setSpeakingMsgId(null);
        fallbackAudioRef.current = null;
      });
    };

    playNext();
  }, []);

  const speakText = useCallback((text, msgId, lang) => {
    // Toggle off — stop any active speech
    if (speakingMsgId === msgId) {
      window.speechSynthesis.cancel();
      if (fallbackAudioRef.current) {
        fallbackAudioRef.current.pause();
        fallbackAudioRef.current = null;
      }
      setSpeakingMsgId(null);
      return;
    }
    window.speechSynthesis.cancel();
    if (fallbackAudioRef.current) {
      fallbackAudioRef.current.pause();
      fallbackAudioRef.current = null;
    }

    const langPrefix = lang ? lang.split('-')[0] : 'en';
    const voices = voicesRef.current.length ? voicesRef.current : window.speechSynthesis.getVoices();

    // Check if a native voice exists for this language
    const exact = lang && voices.find((v) => v.lang.toLowerCase() === lang.toLowerCase());
    const prefix = !exact && lang && voices.find((v) => v.lang.toLowerCase().startsWith(langPrefix));
    const match = exact || prefix;

    if (!match && langPrefix !== 'en') {
      // No native voice available — use Google Translate TTS fallback
      console.log('[TTS] No native voice for', lang, '— using Google Translate TTS fallback');
      speakWithGoogleTTS(text, langPrefix, msgId);
      return;
    }

    // Native TTS
    const utterance = new SpeechSynthesisUtterance(text);
    if (lang) utterance.lang = lang;
    if (match) {
      utterance.voice = match;
      console.log('[TTS] Using native voice:', match.name, match.lang);
    }
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onend = () => setSpeakingMsgId(null);
    utterance.onerror = (e) => {
      console.warn('[TTS] Native speech error:', e.error, '— trying Google Translate fallback');
      // Fallback to Google Translate on error
      if (langPrefix !== 'en') {
        speakWithGoogleTTS(text, langPrefix, msgId);
      } else {
        setSpeakingMsgId(null);
      }
    };

    setSpeakingMsgId(msgId);
    window.speechSynthesis.speak(utterance);
  }, [speakingMsgId, speakWithGoogleTTS]);

  // Cleanup speech on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
      recognitionRef.current?.stop();
      if (fallbackAudioRef.current) { fallbackAudioRef.current.pause(); fallbackAudioRef.current = null; }
    };
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextarea = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';
  };

  const elapsed = () => {
    const mins = Math.round((Date.now() - startTime) / 60000);
    return mins < 1 ? '< 1 min' : `${mins} min`;
  };

  return (
    <motion.div
      key="chat"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="h-screen flex flex-col"
      style={{ background: C.bg }}
    >
      {/* ─── HEADER ─── */}
      <header
        className="flex items-center justify-between px-5 sm:px-6 shrink-0 z-20"
        style={{
          height: 64,
          background: C.surface,
          borderBottom: `1px solid ${C.borderLight}`,
        }}
      >
        <div className="flex items-center gap-3">
          <motion.div
            className="rounded-full"
            style={{ width: 10, height: 10, background: '#22c55e' }}
            animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          />
          <span
            style={{
              fontFamily: font.serif,
              fontSize: 22,
              color: C.heading,
            }}
          >
            {t('brand')}
          </span>
          <span
            style={{
              fontFamily: font.mono,
              fontSize: 11,
              letterSpacing: '0.06em',
              color: C.muted,
              marginLeft: 4,
            }}
          >
            {t('chatOnline')}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <LanguageSelector appLang={appLang} setAppLang={setAppLang} />
          {showSaveBtn && (
            <Button
              variant="outline"
              size="sm"
              className="cursor-pointer"
              onClick={() => setShowSaveDialog(true)}
              style={{
                fontFamily: font.body,
                fontSize: 13,
                fontWeight: 500,
                color: C.body,
                borderRadius: 8,
                borderColor: C.border,
              }}
            >
              {t('saveChat')}
            </Button>
          )}
        </div>
      </header>

      {/* ─── LOCATION PERMISSION BANNER ─── */}
      <AnimatePresence>
        {locationBanner === 'prompt' && (
          <motion.div
            key="loc-prompt"
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="shrink-0 z-10 flex items-center justify-between gap-3 px-5 py-3"
            style={{
              background: C.accentBg,
              borderBottom: `1px solid ${C.accentMuted}`,
              fontFamily: font.body,
              fontSize: 13,
              color: C.body,
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
                <circle cx="12" cy="9" r="2.5"/>
              </svg>
              {t('locPrompt')}
            </span>
            <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
              <button
                onClick={requestLocation}
                style={{
                  fontFamily: font.body, fontSize: 12, fontWeight: 600,
                  background: C.accent, color: '#fff',
                  border: 'none', borderRadius: 6, padding: '5px 14px', cursor: 'pointer',
                }}
              >
                {t('locAllow')}
              </button>
              <button
                onClick={() => setLocationBanner('denied')}
                style={{
                  fontFamily: font.body, fontSize: 12, fontWeight: 400,
                  background: 'transparent', color: C.muted,
                  border: `1px solid ${C.border}`, borderRadius: 6,
                  padding: '5px 12px', cursor: 'pointer',
                }}
              >
                {t('locNotNow')}
              </button>
            </div>
          </motion.div>
        )}

        {locationBanner === 'granted' && (
          <motion.div
            key="loc-granted"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="shrink-0 flex items-center gap-2 px-5 py-2"
            style={{
              background: '#f0fdf4',
              borderBottom: '1px solid #bbf7d0',
              fontFamily: font.body,
              fontSize: 12,
              color: '#15803d',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            {t('locGranted')}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── EMERGENCY ALERT SENT NOTIFICATION ─── */}
      <AnimatePresence>
        {alertSent && (
          <motion.div
            key="alert-sent"
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="shrink-0 flex items-center gap-3 px-5 py-3"
            style={{
              background: '#fff7ed',
              borderBottom: '1px solid #fed7aa',
              fontFamily: font.body,
              fontSize: 13,
              color: '#c2410c',
              fontWeight: 500,
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            {t('alertSent')}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── MESSAGES ─── */}
      <ScrollArea ref={scrollRef} className="flex-1">
        <div
          className="max-w-[680px] mx-auto px-5 sm:px-6"
          style={{ paddingTop: 32, paddingBottom: 16 }}
        >
          <AnimatePresence mode="popLayout">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg.text}
                isBot={msg.isBot}
                isSpeaking={speakingMsgId === msg.id}
                onSpeak={msg.isBot ? () => speakText(msg.text, msg.id, LANG_TO_BCP47[appLang]) : undefined}
                t={t}
              />
            ))}
          </AnimatePresence>

          <QuickChips visible={showChips} onSelect={(chip) => handleSend(chip)} t={t} />

          <AnimatePresence>{isTyping && <TypingIndicator />}</AnimatePresence>

          <AnimatePresence>
            {showTherapistCard && !therapistDone && (
              <TherapistCard
                onClose={() => setShowTherapistCard(false)}
                onDone={() => { setShowTherapistCard(false); setTherapistDone(true); }}
                t={t}
              />
            )}
          </AnimatePresence>
        </div>
      </ScrollArea>

      {/* ─── INPUT BAR ─── */}
      <div
        className="shrink-0 z-20"
        style={{
          background: C.surface,
          borderTop: `1px solid ${C.borderLight}`,
        }}
      >
        <div className="max-w-[680px] mx-auto px-5 sm:px-6 py-4">
          <div
            className="flex items-end gap-2 px-4 py-2"
            style={{
              border: `1px solid ${C.border}`,
              borderRadius: 16,
              background: C.bg,
              transition: 'border-color 0.2s, box-shadow 0.2s',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = C.accent;
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(37,99,235,0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = C.border;
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {/* Mic button */}
            <motion.button
              onClick={toggleListening}
              disabled={crisisActive}
              className="cursor-pointer"
              style={{
                width: 36,
                height: 36,
                borderRadius: 10,
                border: 'none',
                background: isListening ? '#22c55e' : C.surface2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: crisisActive ? 'default' : 'pointer',
                flexShrink: 0,
                marginBottom: 2,
                transition: 'background 0.2s',
              }}
              whileHover={!crisisActive ? { scale: 1.08 } : {}}
              whileTap={!crisisActive ? { scale: 0.92 } : {}}
              title={isListening ? t('micStop') : t('micStart')}
            >
              {isListening
                ? <Mic size={16} style={{ color: '#ffffff' }} />
                : <MicOff size={16} style={{ color: C.muted }} />
              }
            </motion.button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextarea}
              onKeyDown={handleKeyDown}
              disabled={crisisActive}
              placeholder={crisisActive ? t('inputCrisis') : t('inputPlaceholder')}
              rows={1}
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                resize: 'none',
                fontFamily: font.body,
                fontSize: 15,
                fontWeight: 400,
                lineHeight: 1.5,
                color: C.heading,
                padding: '8px 0',
                maxHeight: 140,
              }}
            />
            <motion.button
              onClick={() => handleSend()}
              disabled={!input.trim()}
              className="cursor-pointer"
              style={{
                width: 36,
                height: 36,
                borderRadius: 10,
                border: 'none',
                background: input.trim() ? C.accent : C.surface2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: input.trim() ? 'pointer' : 'default',
                flexShrink: 0,
                marginBottom: 2,
              }}
              whileHover={input.trim() ? { scale: 1.08 } : {}}
              whileTap={input.trim() ? { scale: 0.92 } : {}}
            >
              <ArrowUp
                size={16}
                style={{ color: input.trim() ? '#ffffff' : C.muted }}
              />
            </motion.button>
          </div>
          <p
            style={{
              fontFamily: font.mono,
              fontSize: 10,
              letterSpacing: '0.04em',
              color: C.muted,
              textAlign: 'center',
              marginTop: 10,
            }}
          >
            {t('disclaimer')}
          </p>
        </div>
      </div>

      {/* ─── SAVE DIALOG ─── */}
      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent
          className="p-0 border-0"
          style={{
            background: C.surface,
            border: `1px solid ${C.borderLight}`,
            borderRadius: 16,
            maxWidth: 420,
            boxShadow: '0 24px 48px rgba(0,0,0,0.12)',
          }}
        >
          <div className="p-7">
            <DialogHeader className="mb-0">
              <DialogTitle
                style={{
                  fontFamily: font.serif,
                  fontSize: 28,
                  fontWeight: 400,
                  color: C.heading,
                }}
              >
                {t('saveTitle')}
              </DialogTitle>
              <DialogDescription
                style={{
                  fontFamily: font.body,
                  fontSize: 15,
                  color: C.body,
                  marginTop: 6,
                }}
              >
                {t('saveDesc')}
              </DialogDescription>
            </DialogHeader>

            <Separator className="my-5" style={{ backgroundColor: C.borderLight }} />

            <div
              className="space-y-3 mb-5"
              style={{
                fontFamily: font.mono,
                fontSize: 12,
                letterSpacing: '0.04em',
                color: C.muted,
              }}
            >
              <div className="flex justify-between">
                <span>{t('saveMessages')}</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>{messages.length}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('saveDuration')}</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>{elapsed()}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('saveSession')}</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>{t('saveAnon')}</span>
              </div>
            </div>

            <Separator className="mb-6" style={{ backgroundColor: C.borderLight }} />

            <DialogFooter className="flex flex-col gap-3 sm:flex-col">
              <Button
                onClick={() => setShowSaveDialog(false)}
                className="w-full cursor-pointer"
                style={{
                  fontFamily: font.body,
                  fontSize: 15,
                  fontWeight: 600,
                  background: C.accent,
                  color: '#ffffff',
                  borderRadius: 10,
                  padding: '13px 0',
                }}
              >
                {t('saveCreate')}
              </Button>
              <Button
                variant="ghost"
                onClick={() => setShowSaveDialog(false)}
                className="w-full cursor-pointer"
                style={{
                  fontFamily: font.body,
                  fontSize: 15,
                  fontWeight: 400,
                  color: C.muted,
                  borderRadius: 10,
                }}
              >
                {t('saveEnd')}
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* ─── CRISIS OVERLAY ─── */}
      <AnimatePresence>
        {crisisActive && (
          <CrisisOverlay
            hasLocation={!!userLocation}
            t={t}
            onResume={() => setCrisisActive(false)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   APP ROOT
   ═══════════════════════════════════════════════════════ */
/* ─── LANGUAGE SELECTOR ─── */
const LANG_OPTIONS = [
  { code: 'en', label: 'EN', full: 'English' },
  { code: 'hi', label: 'हि', full: 'हिन्दी' },
  { code: 'kn', label: 'ಕ', full: 'ಕನ್ನಡ' },
];

function LanguageSelector({ appLang, setAppLang }) {
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      <Globe size={14} style={{ color: C.muted, marginRight: 4 }} />
      {LANG_OPTIONS.map((opt) => (
        <button
          key={opt.code}
          onClick={() => setAppLang(opt.code)}
          title={opt.full}
          style={{
            fontFamily: font.body,
            fontSize: 12,
            fontWeight: appLang === opt.code ? 700 : 400,
            color: appLang === opt.code ? '#fff' : C.body,
            background: appLang === opt.code ? C.accent : 'transparent',
            border: appLang === opt.code ? 'none' : `1px solid ${C.border}`,
            borderRadius: 6,
            padding: '4px 10px',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState('landing');
  const [appLang, setAppLang] = useState('en');
  const t = (key) => (translations[appLang] && translations[appLang][key]) || translations.en[key] || key;

  return (
    <AnimatePresence mode="wait">
      {page === 'landing' ? (
        <LandingPage key="landing" onStart={() => setPage('chat')} t={t} appLang={appLang} setAppLang={setAppLang} />
      ) : (
        <ChatPage key="chat" t={t} appLang={appLang} setAppLang={setAppLang} />
      )}
    </AnimatePresence>
  );
}
