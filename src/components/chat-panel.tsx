
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
  onSendMessage: (input: string, temperature: number, file: File | null) => Promise<void>;
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

    await onSendMessage(input, temperature, file);
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
    <div className="flex flex-col h-full bg-gradient-to-br from-background via-background to-muted/20">
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full" viewportRef={scrollAreaViewportRef}>
          <div className="p-6 space-y-6 min-h-full">
            {!isChatSelected ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-full blur-xl"></div>
                  <Button 
                    variant="ghost" 
                    className="relative h-auto p-8 flex flex-col items-center gap-4 hover:bg-gradient-to-r hover:from-primary/10 hover:to-purple-500/10 transition-all duration-300 border border-border/50 rounded-2xl backdrop-blur-sm" 
                    onClick={onNewChat}
                  >
                    <div className="p-4 bg-gradient-to-r from-primary to-purple-500 rounded-full">
                      <MessageSquarePlus size={32} className="text-white" />
                    </div>
                    <div className="space-y-2">
                      <p className="text-xl font-semibold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                        Start a new conversation
                      </p>
                      <p className="text-sm text-muted-foreground">Click here to begin your journey</p>
                    </div>
                  </Button>
                </div>
              </div>
            ) : messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-full blur-xl"></div>
                  <div className="relative p-6 bg-card/50 backdrop-blur-sm rounded-2xl border border-border/50">
                    <div className="p-4 bg-gradient-to-r from-primary to-purple-500 rounded-full w-fit mx-auto mb-4">
                      <Bot className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent mb-2">
                      Aether-Enhanced Golem
                    </h3>
                    <p className="text-muted-foreground">Ask me anything, or attach a file to begin your mystical journey</p>
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
        <div className="border-t border-border/50 bg-card/50 backdrop-blur-sm">
          <div className="p-4 space-y-3">
            {file && (
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20">
                  <Paperclip className="h-3 w-3 mr-1" />
                  {file.name}
                </Badge>
                <Button variant="ghost" size="icon" className="h-6 w-6 hover:bg-destructive/10 hover:text-destructive" onClick={() => setFile(null)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            )}
            <form onSubmit={handleSubmit} className="flex w-full items-end gap-3">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="shrink-0 h-10 w-10 rounded-full hover:bg-primary/10 hover:text-primary transition-all duration-200"
                aria-label="Attach file"
              >
                <Paperclip className="h-5 w-5" />
              </Button>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
              <div className="flex-1 relative">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message or attach a file..."
                  className="flex-1 resize-none min-h-[44px] max-h-40 bg-background/50 backdrop-blur-sm border-border/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary/20 focus:border-primary/50 transition-all duration-200"
                  rows={1}
                  disabled={isLoading}
                />
              </div>
              <div className='flex items-center gap-2'>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className={cn(
                          "h-10 w-10 rounded-full transition-all duration-200",
                          golemActivated 
                            ? 'bg-gradient-to-r from-primary to-purple-500 text-white shadow-lg' 
                            : 'hover:bg-primary/10 hover:text-primary'
                        )}
                      >
                        <BrainCircuit className="h-5 w-5" />
                        <span className="sr-only">Golem Configurations</span>
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-96 bg-card/95 backdrop-blur-sm border-border/50">
                      <Tabs defaultValue="activation" className="w-full">
                        <TabsList className="grid w-full grid-cols-3 bg-muted/50">
                          <TabsTrigger value="activation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Activation</TabsTrigger>
                          <TabsTrigger value="sefirot" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Sefirot</TabsTrigger>
                          <TabsTrigger value="model" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Model</TabsTrigger>
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
                                <h4 className="font-medium leading-none">Model Parameters</h4>
                                <p className="text-sm text-muted-foreground">
                                  Adjust core generation parameters.
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
                        </TabsContent>
                      </Tabs>
                    </PopoverContent>
                  </Popover>

                  <Button
                    type="submit"
                    disabled={(!input.trim() && !file) || isLoading}
                    className="h-10 w-10 rounded-full bg-gradient-to-r from-primary to-purple-500 hover:from-primary/90 hover:to-purple-500/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg"
                  >
                    <SendHorizonal className="h-5 w-5" />
                    <span className="sr-only">Send message</span>
                  </Button>
                </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
