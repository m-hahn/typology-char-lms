load("/Users/lscpuser/Downloads/segmented_acqdiv_corpus_2018-08-27.Rdata")
load("/Users/lscpuser/Downloads/acqdiv_corpus_2018-08-27.rda")

head(segmented.utterances)
summary(sessions_assigned)

#Yucatec (sessions_assigned:2916-3144  vs utterances: 2916-3149, 92963), Turkish (sessions_assigned: 2695-2788 vs utterances: 2543-2915) not the same

languages<-c(  "Chintang")

for (lang in languages){

utterances_lang=utterances[utterances$language==lang,]
speakers_lang=speakers[speakers$language==lang,]
morphemes_lang=morphemes[morphemes$language==lang,]
words_lang=words[words$language==lang,]
sessions_lang=sessions[sessions$language==lang,]
uniquespeakers_lang=uniquespeakers[uniquespeakers$language==lang,]

write.table(speakers_lang, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/speakers.csv"), row.names=F, col.names=T, sep="#", quote=F)
write.table(morphemes_lang, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/morphemes.csv"), row.names=F, col.names=T, sep="#", quote=F)
write.table(words_lang, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/words.csv"), row.names=F, col.names=T, sep="#", quote=F)
write.table(sessions_lang, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/sessions.csv"), row.names=F, col.names=T, sep="#", quote=F)
write.table(uniquespeakers_lang, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/uniquespeakers.csv"), row.names=F, col.names=T, sep="#", quote=F)



sessions_assigned_lang=sessions_assigned[sessions_assigned$language==lang,]

colnames(utterances_lang)[colnames(utterances_lang)=="session_id_fk"] <- "session_id"
merge(utterances_lang,sessions_assigned_lang,by=c("session_id", "language"))->main_lang

main_lang$translation<-NULL
main_lang$comment<-NULL
main_lang$warning<-NULL

merge(main_lang, segmented.utterances,by=c("utterance_id"))->main_lang_segmented

selcol="segmented.utterance"
x=main_lang_segmented[selcol]

toremove=c(":","^","'","(",")","&","?",".",",","=","…","!","_","/","।","«","‡","§","™","•","�","Œ","£","±","-","ǃ", "&ADV","&CAUS","&COND","&CONN","&NEG","&IMP","&PAST","&POL","&PRES","&QUOT","&SGER","-HON_","_NEG", ".*FS_","FS_", "DELAY", "\340\245\207", "\314\265","\314\200") 
for(thiscar in toremove) x<-gsub(thiscar,"",x,fixed=T)

main_lang_segmented[grep("NA",as.character(main_lang_segmented[,"segmented.utterance"]),invert=T),]->main_lang_segmented 
print(summary(main_Chintang_segmented))
names=c("utterance_id", "session_id", "language", "speaker_id_fk", "utterance_morphemes", "assign", "segmented.utterance")
main_lang_segmented1<-main_lang_segmented[names]



corpus_split<-c("dev", "train", "test")
for (split in corpus_split){
main_lang_segmented_split=main_lang_segmented1[main_lang_segmented$assign==split,]
write.table(main_lang_segmented_split, file=paste0("/Users/lscpuser/Documents/fair_project/acqdiv_final_data/", lang, "/utterances_", split, ".csv"), row.names=F, col.names=T, sep="#", quote=F)

 }
}  
  







