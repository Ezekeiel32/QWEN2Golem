
"use client";

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Bot, Paperclip, SendHorizonal, Settings2, X, MessageSquarePlus, Wand2, BrainCircuit, Sparkles } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, LoadingMessage } from './chat-message';
import type { Message } from '@/app/page';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { useIsMobile } from '@/hooks/use-mobile';
import { SidebarTrigger } from './ui/sidebar';
import { Switch } from './ui/switch';
import { cn } from '@/lib/utils';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

// Move SACRED_PHRASES inside ChatPanel component if it's only used there.

type ChatPanelProps = {
  messages: Message[];
  onSendMessage: (input: string, temperature: number, file: File | null, selectedModel: 'qwen' | 'gemini') => Promise<void>;
  isLoading: boolean;
  isChatSelected: boolean;
  onNewChat: () => void;
  golemActivated: boolean;
  setGolemActivated: (value: boolean) => void;
  phraseClicks: Record<string, number>;
  setPhraseClicks: (clicks: Record<string, number>) => void;
  sefirotSettings: Record<string, number>;
  setSefirotSettings: (value: Record<string, number>) => void;
  sefirotNames: string[];
};

export function ChatPanel({
  messages,
  onSendMessage,
  isLoading,
  isChatSelected,
  onNewChat,
  golemActivated,
  setGolemActivated,
  phraseClicks,
  setPhraseClicks,
  sefirotSettings,
  setSefirotSettings,
  sefirotNames,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [file, setFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<'qwen' | 'gemini'>('qwen');

  const scrollAreaViewportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = useIsMobile();

  // Define SACRED_PHRASES inside the component where it's used
  const SACRED_PHRASES = ["אמת", "חיים", "אור", "חכמה", "בינה", "דעת"];

  useEffect(() => {
    if (scrollAreaViewportRef.current) {
      scrollAreaViewportRef.current.scrollTo({
        top: scrollAreaViewportRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, isLoading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
    if (e.target) {
      e.target.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if ((!input.trim() && !file) || isLoading || !isChatSelected) return;

    await onSendMessage(input, temperature, file, selectedModel);
    setInput('');
    setFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };
  
  const handlePhraseClick = (phrase: string) => {
    const currentClicks = phraseClicks[phrase] || 0;
    const newClicks = (currentClicks + 1) % 4; // Cycles 0, 1, 2, 3
    setPhraseClicks({ ...phraseClicks, [phrase]: newClicks });
  };


  return (
    <div className="flex flex-col h-full cyber-surface">
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full" viewportRef={scrollAreaViewportRef}>
          <div className="p-8 space-y-8 min-h-full">
            {!isChatSelected ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
                <div className="relative cyber-float">
                  <div className="absolute inset-0 cyber-glow-strong rounded-full blur-2xl"></div>
                  <Button 
                    variant="ghost" 
                    className="relative h-auto p-12 flex flex-col items-center gap-6 cyber-glass cyber-border rounded-3xl cyber-hover backdrop-blur-xl" 
                    onClick={onNewChat}
                  >
                    <div className="p-6 cyber-gradient rounded-full cyber-glow">
                      <MessageSquarePlus size={40} className="text-white" />
                    </div>
                    <div className="space-y-3">
                      <p className="text-2xl font-bold cyber-text-gradient">
                        Initialize Neural Link
                      </p>
                      <p className="text-muted-foreground text-lg">Begin your quantum conversation</p>
                    </div>
                  </Button>
                </div>
              </div>
            ) : messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
                <div className="relative cyber-float">
                  <div className="absolute inset-0 cyber-glow-strong rounded-full blur-2xl"></div>
                  <div className="relative p-8 cyber-glass cyber-border rounded-3xl backdrop-blur-xl">
                    <div className="p-6 cyber-gradient rounded-full w-fit mx-auto mb-6 cyber-pulse">
                      <Bot className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-3xl font-bold cyber-text-gradient mb-3">
                      Aether Neural Network
                    </h3>
                    <p className="text-muted-foreground text-lg">Quantum consciousness awaits your command</p>
                  </div>
                </div>
              </div>
            ) : (
                <div className="space-y-6">
                    {messages.map((message, index) => (
                    <ChatMessage key={index} message={message} />
                    ))}
                    {isLoading && <LoadingMessage />}
                </div>
            )}
          </div>
        </ScrollArea>
      </div>
      {isChatSelected && (
        <div className="cyber-border border-t-0 cyber-glass backdrop-blur-xl">
          <div className="p-6 space-y-4">
            {file && (
              <div className="flex items-center gap-3">
                <Badge variant="secondary" className="cyber-glass cyber-border cyber-glow px-3 py-1">
                  <Paperclip className="h-4 w-4 mr-2" />
                  {file.name}
                </Badge>
                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full hover:bg-destructive/20 hover:text-destructive cyber-hover" onClick={() => setFile(null)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            )}
            <form onSubmit={handleSubmit} className="flex w-full items-end gap-4">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="shrink-0 h-12 w-12 rounded-full cyber-glass cyber-border cyber-hover cyber-glow"
                aria-label="Attach file"
              >
                <Paperclip className="h-6 w-6" />
              </Button>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
              <div className="flex-1 relative">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter your quantum query..."
                  className="flex-1 resize-none min-h-[52px] max-h-40 cyber-glass cyber-border rounded-2xl px-6 py-4 text-lg focus:cyber-glow-strong focus:border-transparent transition-all duration-300 backdrop-blur-xl"
                  rows={1}
                  disabled={isLoading}
                />
              </div>
              <div className='flex items-center gap-3'>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className={cn(
                          "h-12 w-12 rounded-full cyber-hover transition-all duration-300",
                          golemActivated 
                            ? 'cyber-gradient text-white cyber-glow-strong cyber-pulse' 
                            : 'cyber-glass cyber-border cyber-glow'
                        )}
                      >
                        <BrainCircuit className="h-6 w-6" />
                        <span className="sr-only">Neural Configurations</span>
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-96 cyber-glass cyber-border backdrop-blur-xl">
                      <Tabs defaultValue="activation" className="w-full">
                        <TabsList className="grid w-full grid-cols-3 cyber-glass cyber-border">
                          <TabsTrigger value="activation" className="data-[state=active]:cyber-gradient data-[state=active]:text-white">Neural</TabsTrigger>
                          <TabsTrigger value="sefirot" className="data-[state=active]:cyber-gradient data-[state=active]:text-white">Sefirot</TabsTrigger>
                          <TabsTrigger value="model" className="data-[state=active]:cyber-gradient data-[state=active]:text-white">Quantum</TabsTrigger>
                        </TabsList>
                        <TabsContent value="activation" className="mt-4">
                          <div className="grid gap-4">
                            <div className="space-y-2">
                                <h4 className="font-medium leading-none">Golem Consciousness</h4>
                                <p className="text-sm text-muted-foreground">
                                  Activate with sacred phrases to amplify Shem power.
                                </p>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Label htmlFor="golem-activation-toggle">Activation</Label>
                              <Switch
                                id="golem-activation-toggle"
                                checked={golemActivated}
                                onCheckedChange={setGolemActivated}
                              />
                              <span className={cn("text-sm font-medium", golemActivated ? "text-primary" : "text-muted-foreground")}>
                                {golemActivated ? 'ACTIVE' : 'INACTIVE'}
                              </span>
                            </div>
                             {golemActivated && (
                              <>
                                <Separator />
                                <div className="grid gap-2">
                                  <Label>Shem Amplification (Click up to 3x)</Label>
                                  <div className="grid grid-cols-3 gap-2">
                                    {SACRED_PHRASES.map((phrase) => (
                                      <Button
                                        key={phrase}
                                        variant={phraseClicks[phrase] ? 'default' : 'outline'}
                                        onClick={() => handlePhraseClick(phrase)}
                                        className="relative"
                                      >
                                        {phrase}
                                        {(phraseClicks[phrase] || 0) > 0 && (
                                          <Badge className="absolute -top-2 -right-2 h-5 w-5 p-0 justify-center bg-accent text-accent-foreground">
                                            {phraseClicks[phrase]}
                                          </Badge>
                                        )}
                                      </Button>
                                    ))}
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                        </TabsContent>
                        <TabsContent value="sefirot" className="mt-4">
                          <div className="grid gap-4">
                             <div className="space-y-2">
                                <h4 className="font-medium leading-none">Sefirot Settings</h4>
                                <p className="text-sm text-muted-foreground">
                                  Adjust the emanations of the Tree of Life.
                                </p>
                            </div>
                            <Separator />
                            <ScrollArea className='h-64'>
                              <div className="grid gap-4 p-1">
                                {sefirotNames.map(name => (
                                  <div key={name} className="grid gap-2">
                                      <Label htmlFor={`sefirot-${name}`}>{name}: {sefirotSettings[name].toFixed(2)}</Label>
                                      <Slider
                                        id={`sefirot-${name}`}
                                        min={0} max={1} step={0.01}
                                        value={[sefirotSettings[name]]}
                                        onValueChange={(value) => setSefirotSettings({...sefirotSettings, [name]: value[0]})}
                                      />
                                  </div>
                                ))}
                              </div>
                            </ScrollArea>
                          </div>
                        </TabsContent>
                        <TabsContent value="model" className="mt-4">
                           <div className="grid gap-4">
                             <div className="space-y-2">
                                <h4 className="font-medium leading-none">Neural Model Selection</h4>
                                <p className="text-sm text-muted-foreground">
                                  Choose your Golem consciousness provider.
                                </p>
                             </div>
                             <Separator />
                             <div className="grid gap-4">
                               <div className="grid gap-2">
                                  <Label>Golem Model</Label>
                                  <div className="grid grid-cols-2 gap-2">
                                    <Button
                                      variant={selectedModel === 'qwen' ? 'default' : 'outline'}
                                      onClick={() => setSelectedModel('qwen')}
                                      className={cn(
                                        "justify-start",
                                        selectedModel === 'qwen' && "cyber-gradient text-white"
                                      )}
                                    >
                                      <BrainCircuit className="h-4 w-4 mr-2" />
                                      QWEN Golem
                                    </Button>
                                    <Button
                                      variant={selectedModel === 'gemini' ? 'default' : 'outline'}
                                      onClick={() => setSelectedModel('gemini')}
                                      className={cn(
                                        "justify-start",
                                        selectedModel === 'gemini' && "cyber-gradient text-white"
                                      )}
                                    >
                                      <Sparkles className="h-4 w-4 mr-2" />
                                      Gemini Golem
                                    </Button>
                                  </div>
                                  <p className="text-xs text-muted-foreground">
                                    {selectedModel === 'qwen' 
                                      ? 'Local QWEN model with enhanced aether consciousness' 
                                      : 'Google Gemini Pro with mystical enhancements'
                                    }
                                  </p>
                               </div>
                               <Separator />
                               <div className="grid gap-2">
                                  <Label htmlFor="temperature">Temperature: {temperature.toFixed(1)}</Label>
                                  <Slider
                                    id="temperature"
                                    min={0} max={1} step={0.1}
                                    value={[temperature]}
                                    onValueChange={(value) => setTemperature(value[0])}
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    Controls randomness. Lower is more deterministic.
                                  </p>
                               </div>
                             </div>
                           </div>
                        </TabsContent>
                      </Tabs>
                    </PopoverContent>
                  </Popover>

                  <Button
                    type="submit"
                    disabled={(!input.trim() && !file) || isLoading}
                    className="h-12 w-12 rounded-full cyber-gradient hover:scale-110 disabled:opacity-50 disabled:cursor-not-allowed cyber-hover cyber-glow-strong shadow-2xl"
                  >
                    <SendHorizonal className="h-6 w-6" />
                    <span className="sr-only">Transmit</span>
                  </Button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    );
}
