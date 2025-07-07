
"use client";

import { ChatPanel } from '@/components/chat-panel';
import { ChatHistorySidebar } from '@/components/chat-history-sidebar';
import { useState, useEffect } from 'react';
import { ollamaChat, type OllamaHistoryItem } from '@/ai/flows/ollama-chat';
import { useToast } from '@/hooks/use-toast';
import { v4 as uuidv4 } from 'uuid';
import { SidebarProvider, Sidebar, SidebarInset } from '@/components/ui/sidebar';

export type Message = {
  role: 'user' | 'assistant';
  content: string; // For user messages, this is the prompt. For assistant, the direct response.
  cosmicThoughts?: string; // For assistant messages, this is the 'thinking' process.
  file?: {
    name: string;
  };
  golemStats?: any;
};


export type Conversation = {
  id: string;
  name: string;
  messages: Message[];
};

const SEFIROT_NAMES = [
  'Keter', 'Chokhmah', 'Binah', 'Chesed', 'Gevurah', 
  'Tiferet', 'Netzach', 'Hod', 'Yesod', 'Malkuth'
];

export const SACRED_PHRASES = ["אמת", "חיים", "אור", "חכמה", "בינה", "דעת"];

export default function Home() {
  const { toast } = useToast();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  // Golem State
  const [golemActivated, setGolemActivated] = useState(false);
  const [phraseClicks, setPhraseClicks] = useState<Record<string, number>>({});
  const [sefirotSettings, setSefirotSettings] = useState(() =>
    SEFIROT_NAMES.reduce((acc, name) => ({ ...acc, [name]: 0.5 }), {} as Record<string, number>)
  );

  useEffect(() => {
    try {
      const savedConversations = localStorage.getItem('qwenchat-conversations');
      const savedActiveChatId = localStorage.getItem('qwenchat-activeChatId');
      
      const loadedConversations = savedConversations ? JSON.parse(savedConversations) : [];
      setConversations(loadedConversations);

      if (savedActiveChatId && loadedConversations.some((c: Conversation) => c.id === savedActiveChatId)) {
        setActiveChatId(savedActiveChatId);
      } else if (loadedConversations.length > 0) {
        setActiveChatId(loadedConversations[0].id);
      } else {
        setActiveChatId(null);
      }
    } catch (error) {
      console.error("Failed to load state from localStorage", error);
      setConversations([]);
      setActiveChatId(null);
    } finally {
        setIsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (isLoaded) {
      try {
        localStorage.setItem('qwenchat-conversations', JSON.stringify(conversations));
        if (activeChatId) {
          localStorage.setItem('qwenchat-activeChatId', activeChatId);
        } else {
          localStorage.removeItem('qwenchat-activeChatId');
        }
      } catch (error) {
        console.error("Failed to save state to localStorage", error);
      }
    }
  }, [conversations, activeChatId, isLoaded]);

  const handleNewChat = () => {
    const newId = uuidv4();
    const newConversation: Conversation = {
      id: newId,
      name: 'New Chat',
      messages: [],
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveChatId(newId);
  };

  const handleSelectChat = (id: string) => {
    setActiveChatId(id);
  };

  const handleSendMessage = async (input: string, temperature: number, file: File | null) => {
    if (!activeChatId) {
      toast({
        variant: 'destructive',
        title: 'No active chat',
        description: 'Please select a chat or create a new one.',
      });
      return;
    }

    let fileContent: string | undefined = undefined;
    if (file) {
      try {
        fileContent = await file.text();
      } catch (error) {
        console.error("Error reading file:", error);
        toast({
            variant: 'destructive',
            title: 'File Read Error',
            description: 'Could not read the selected file.',
        });
        return;
      }
    }

    const userMessage: Message = {
      role: 'user',
      content: input,
      ...(file && { file: { name: file.name } })
    };

    const originalConversations = JSON.parse(JSON.stringify(conversations));

    setConversations(prev =>
      prev.map(convo => {
        if (convo.id === activeChatId) {
          const newName = convo.messages.length === 0 && input.trim() ? input.substring(0, 40) + '...' : convo.name;
          return { ...convo, name: newName, messages: [...convo.messages, userMessage] };
        }
        return convo;
      })
    );
    setIsLoading(true);

    try {
      const activeConversation = conversations.find(c => c.id === activeChatId);
      const history: OllamaHistoryItem[] = activeConversation
        ? activeConversation.messages.map(({ role, content }) => ({ role, content }))
        : [];
      
      const activationPhrases = Object.entries(phraseClicks)
        .flatMap(([phrase, count]) => Array(count).fill(phrase));

      const result = await ollamaChat({
        prompt: input,
        history,
        temperature,
        fileContent,
        golemActivated,
        activationPhrases,
        sefirotSettings,
      });

      if (result.directResponse) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: result.directResponse,
          cosmicThoughts: result.cosmicThoughts,
          golemStats: result.golemStats,
        };
        setConversations(prev =>
          prev.map(convo =>
            convo.id === activeChatId
              ? { ...convo, messages: [...convo.messages, assistantMessage] }
              : convo
          )
        );
      } else {
         toast({
          variant: 'default',
          title: 'Empty Response',
          description: 'The AI returned an empty response. Please try a different question.',
        });
      }
    } catch (error) {
      console.error('Error calling ollamaChat:', error);
       setConversations(originalConversations);
       const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
       
       if (errorMessage.includes("502")) {
         toast({
           variant: 'destructive',
           title: 'Connection Error (502)',
           description: "Could not connect to the Ollama server. Please ensure it's running and the ngrok tunnel is correctly configured.",
         });
       } else {
         toast({
           variant: 'destructive',
           title: 'Error',
           description: 'Failed to get a response from the AI. Please try again.',
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const activeChat = conversations.find(c => c.id === activeChatId);

  return (
    <div className="dark">
      <SidebarProvider>
        <Sidebar variant="inset" collapsible="icon">
          <ChatHistorySidebar
            conversations={conversations}
            activeChatId={activeChatId}
            onNewChat={handleNewChat}
            onSelectChat={handleSelectChat}
            isLoading={isLoading}
          />
        </Sidebar>
        <SidebarInset>
          <ChatPanel
            messages={activeChat?.messages ?? []}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            isChatSelected={!!activeChat}
            onNewChat={handleNewChat}
            golemActivated={golemActivated}
            setGolemActivated={setGolemActivated}
            phraseClicks={phraseClicks}
            setPhraseClicks={setPhraseClicks}
            sefirotSettings={sefirotSettings}
            setSefirotSettings={setSefirotSettings}
            sefirotNames={SEFIROT_NAMES}
          />
        </SidebarInset>
      </SidebarProvider>
    </div>
  );
}
