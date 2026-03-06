import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { ArrowUp, Shield, Lock, Eye, Users, ArrowRight } from 'lucide-react';
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
function MessageBubble({ message, isBot }) {
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
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   QUICK REPLY CHIPS
   ═══════════════════════════════════════════════════════ */
function QuickChips({ onSelect, visible }) {
  const chips = [
    'I needed to vent',
    'Feeling anxious',
    'Having a rough day',
    'Not sure where to start',
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
function LandingPage({ onStart }) {
  const features = [
    {
      icon: Lock,
      title: 'HIPAA-Grade Privacy',
      desc: 'Your data is encrypted end-to-end. We follow HIPAA, SOC 2, and GDPR standards to keep every conversation secure.',
    },
    {
      icon: Shield,
      title: 'Fully Anonymous',
      desc: 'No account, no email, no phone number. Start talking with zero personal information required.',
    },
    {
      icon: Eye,
      title: 'Your Data, Your Control',
      desc: 'We never sell or share your data with third parties. You can delete your conversation history at any time.',
    },
    {
      icon: Users,
      title: 'Human Escalation',
      desc: 'If the conversation indicates a crisis, we connect you to real crisis counselors and emergency services.',
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
          Solace
        </span>
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
          Talk to us
        </Button>
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
            You deserve to be
            <br />
            <span style={{ fontStyle: 'italic', color: C.accent }}>heard.</span>
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
          A safe, private space to talk about what you are feeling.
          Solace listens, understands, and is here for you — always.
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
            Talk to Solace
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
          Free and anonymous. No sign-up needed.
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
              cx="10" cy="10" r="3" fill={C.accent}
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
              It takes{' '}
              <span style={{ fontStyle: 'italic', color: C.accent }}>strength</span>{' '}
              to reach out
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
              You are not broken. You are not too much. Whatever you are going through
              right now — stress, anxiety, sadness — you do not have to face it alone.
              Solace is here to listen, without judgment, whenever you need it.
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
                Your Safety Comes First
              </span>
              <h2
                style={{
                  fontFamily: font.serif,
                  fontSize: 'clamp(28px, 4.5vw, 40px)',
                  fontWeight: 400,
                  color: C.heading,
                }}
              >
                Built on{' '}
                <span style={{ fontStyle: 'italic', color: C.accent }}>trust</span>
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
                Simple by Design
              </span>
              <h2
                style={{
                  fontFamily: font.serif,
                  fontSize: 'clamp(28px, 4.5vw, 40px)',
                  fontWeight: 400,
                  color: C.heading,
                }}
              >
                Three steps. That is it.
              </h2>
            </div>
          </Reveal>

          {[
            { num: '01', title: 'Start talking', desc: 'Click the button. No account, no forms, no barriers. You begin immediately.' },
            { num: '02', title: 'Solace listens', desc: 'Our AI understands how you are feeling and responds with genuine empathy and care.' },
            { num: '03', title: 'You decide', desc: 'Save the conversation to come back later, or leave. It is entirely up to you.' },
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
              You have already taken
              <br />
              the{' '}
              <span style={{ fontStyle: 'italic', color: C.accent }}>hardest</span>{' '}
              step
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
              Acknowledging that something feels off is brave.
              Let Solace meet you where you are.
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
              Talk to Solace
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
          SOLACE IS NOT A REPLACEMENT FOR PROFESSIONAL THERAPY.
          IF YOU ARE IN CRISIS, CONTACT EMERGENCY SERVICES.
        </p>
      </footer>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   CHAT PAGE
   ═══════════════════════════════════════════════════════ */
function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showChips, setShowChips] = useState(false);
  const [userMsgCount, setUserMsgCount] = useState(0);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showSaveBtn, setShowSaveBtn] = useState(false);
  const [startTime] = useState(Date.now());
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const hasSaveTriggered = useRef(false);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      const el = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (el) el.scrollTop = el.scrollHeight;
    }
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
        text: "Hey, I am glad you are here. There is no right or wrong way to start \u2014 just share whatever is on your mind.",
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
        text: "Take your time. I am not going anywhere.",
        isBot: true,
        time: now(),
      }]);
    }, 500 + 1200 + 600 + 1400);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); };
  }, []);

  const botResponses = [
    "That sounds like it has been weighing on you. How long have you been feeling this way?",
    "I hear you. It takes real honesty to put those feelings into words.",
    "That makes sense given what you are going through. You are not overreacting.",
    "I want you to know that whatever you are feeling right now is valid.",
    "Sometimes naming the feeling is the hardest part. You have already done that.",
    "It sounds like you have been carrying a lot. You do not have to sort through it all at once.",
    "Thank you for trusting me with that. Can you tell me more about when this started?",
    "That resonates. A lot of people feel that way but never say it \u2014 the fact that you are here matters.",
  ];

  const handleSend = useCallback(
    (text) => {
      const trimmed = (text || input).trim();
      if (!trimmed) return;

      setMessages((prev) => [...prev, {
        id: Date.now(),
        text: trimmed,
        isBot: false,
        time: now(),
      }]);
      setInput('');
      setShowChips(false);

      const newCount = userMsgCount + 1;
      setUserMsgCount(newCount);
      if (newCount >= 5) setShowSaveBtn(true);

      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }

      const typingDelay = 400;
      const typingDuration = 800 + Math.random() * 500;

      setTimeout(() => setIsTyping(true), typingDelay);
      setTimeout(() => {
        setIsTyping(false);

        if (newCount >= 5 && !hasSaveTriggered.current) {
          hasSaveTriggered.current = true;
          setMessages((prev) => [...prev, {
            id: Date.now() + 1,
            text: "We have been talking for a while. If you would like to continue this tomorrow, you can save this conversation.",
            isBot: true,
            time: now(),
          }]);
          setTimeout(() => setShowSaveDialog(true), 1200);
        } else {
          const idx = (newCount - 1) % botResponses.length;
          setMessages((prev) => [...prev, {
            id: Date.now() + 1,
            text: botResponses[idx],
            isBot: true,
            time: now(),
          }]);
        }
      }, typingDelay + typingDuration);
    },
    [input, userMsgCount, botResponses]
  );

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
            Solace
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
            online
          </span>
        </div>
        <div>
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
              Save chat
            </Button>
          )}
        </div>
      </header>

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
              />
            ))}
          </AnimatePresence>

          <QuickChips visible={showChips} onSelect={(chip) => handleSend(chip)} />

          <AnimatePresence>{isTyping && <TypingIndicator />}</AnimatePresence>
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
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextarea}
              onKeyDown={handleKeyDown}
              placeholder="Say anything..."
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
            Not a crisis service — call emergency services if you are in danger
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
                Save this session
              </DialogTitle>
              <DialogDescription
                style={{
                  fontFamily: font.body,
                  fontSize: 15,
                  color: C.body,
                  marginTop: 6,
                }}
              >
                Pick up where you left off next time.
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
                <span>Messages</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>{messages.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Duration</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>{elapsed()}</span>
              </div>
              <div className="flex justify-between">
                <span>Session</span>
                <span style={{ color: C.heading, fontWeight: 500 }}>Anonymous</span>
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
                Create account to save
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
                End session
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   APP ROOT
   ═══════════════════════════════════════════════════════ */
export default function App() {
  const [page, setPage] = useState('landing');

  return (
    <AnimatePresence mode="wait">
      {page === 'landing' ? (
        <LandingPage key="landing" onStart={() => setPage('chat')} />
      ) : (
        <ChatPage key="chat" />
      )}
    </AnimatePresence>
  );
}
