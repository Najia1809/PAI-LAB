import nltk
from nltk.chat.util import Chat, reflections
from flask import Flask, render_template, request, jsonify

nltk.download('punkt')

pairs = [
    [r"Hello|Hi|Hey|hello|hi|hey",
     ["Hello! Welcome to Grand Palace Hotel! How can I assist you?",
      "Hi there! How can I help you with your stay at Grand Palace Hotel?",
      "Hey! Welcome! What can I do for you today?"]],

    [r".*room types.*|.*types of rooms.*|.*available rooms.*|.*what rooms.*",
     ["We have Standard Rooms, Deluxe Rooms, Executive Suites, and Presidential Suites available.",
      "Our hotel offers Standard, Deluxe, Executive Suite and Presidential Suite rooms for your comfort.",
      "We offer a variety of rooms: Standard, Deluxe, Executive Suite, and Presidential Suite."]],

    [r".*price.*|.*cost.*|.*charges.*|.*how much.*|.*rates.*",
     ["Our room rates are: Standard Room: $80/night, Deluxe Room: $150/night, Executive Suite: $250/night, Presidential Suite: $500/night.",
      "Prices start from $80 per night for Standard rooms up to $500 per night for our Presidential Suite.",
      "Standard: $80/night | Deluxe: $150/night | Executive Suite: $250/night | Presidential Suite: $500/night."]],

    [r".*amenities.*|.*facilities.*|.*services.*|.*what do you offer.*",
     ["We offer free WiFi, swimming pool, gym, spa, restaurant, room service, airport transfer and conference halls.",
      "Our amenities include: Free WiFi, Swimming Pool, Gym, Spa, Restaurant, 24/7 Room Service and Airport Transfer.",
      "Grand Palace Hotel provides WiFi, Pool, Gym, Spa, Restaurant, Room Service and much more!"]],

    [r".*book.*|.*reservation.*|.*reserve.*|.*how to book.*",
     ["You can book a room by calling us at +1-800-GRAND or visiting our website at www.grandpalacehotel.com.",
      "To make a reservation, visit our website or call our front desk at +1-800-GRAND. We are available 24/7!",
      "Booking is easy! Visit www.grandpalacehotel.com or call +1-800-GRAND to reserve your room today."]],

    [r".*check.?in.*|.*check in.*|.*checkin.*",
     ["Our check-in time is 2:00 PM. Early check-in is available upon request based on room availability.",
      "Check-in starts at 2:00 PM. If you need early check-in, please contact our front desk in advance.",
      "Standard check-in time is 2:00 PM. Early check-in may be arranged for an additional charge."]],

    [r".*check.?out.*|.*check out.*|.*checkout.*",
     ["Check-out time is 12:00 PM noon. Late check-out is available upon request.",
      "You need to check out by 12:00 PM. If you need extra time, please inform the front desk.",
      "Our check-out time is 12:00 PM. Late check-out requests are subject to availability."]],

    [r".*wifi.*|.*internet.*|.*wi-fi.*",
     ["Yes! We offer free high-speed WiFi throughout the hotel for all guests.",
      "Complimentary WiFi is available in all rooms and public areas of the hotel.",
      "Free WiFi is available for all guests. Ask our front desk for the password upon check-in."]],

    [r".*pool.*|.*swimming.*",
     ["Our outdoor swimming pool is open from 7:00 AM to 10:00 PM daily for all guests.",
      "Yes we have a beautiful outdoor pool available from 7 AM to 10 PM. Enjoy your swim!",
      "The swimming pool is open daily from 7:00 AM to 10:00 PM. Towels are provided free of charge."]],

    [r".*restaurant.*|.*food.*|.*dining.*|.*breakfast.*",
     ["Our in-house restaurant serves breakfast from 7 AM, lunch from 12 PM, and dinner from 7 PM.",
      "Grand Palace Restaurant is open all day. We serve continental breakfast, lunch and dinner.",
      "We have a restaurant serving delicious meals from 7:00 AM to 11:00 PM. Room service is also available 24/7."]],

    [r".*gym.*|.*fitness.*|.*workout.*",
     ["Our fully equipped gym is open 24/7 for all guests. Personal trainers are available on request.",
      "Yes we have a modern fitness center open 24 hours a day for all guests.",
      "The gym is available 24/7. Equipment includes treadmills, weights, and yoga mats."]],

    [r".*spa.*|.*massage.*|.*salon.*",
     ["Our luxury spa offers massages, facials, and beauty treatments. Open from 9 AM to 9 PM.",
      "The spa is open from 9:00 AM to 9:00 PM. We offer relaxing massages and beauty treatments.",
      "Yes we have a full spa! Services include Swedish massage, hot stone therapy, and facial treatments."]],

    [r".*location.*|.*address.*|.*where.*|.*how to reach.*",
     ["Grand Palace Hotel is located at 123 Luxury Avenue, Downtown City. 5 minutes from the airport.",
      "We are at 123 Luxury Avenue, Downtown. Easily accessible by taxi, bus or airport transfer.",
      "Our address is 123 Luxury Avenue, Downtown City. We provide free airport transfer for our guests."]],

    [r".*contact.*|.*phone.*|.*email.*|.*number.*",
     ["You can reach us at +1-800-GRAND or email us at info@grandpalacehotel.com.",
      "Contact us at +1-800-GRAND (24/7) or send an email to info@grandpalacehotel.com.",
      "Phone: +1-800-GRAND | Email: info@grandpalacehotel.com | We are available 24/7!"]],

    [r".*cancel.*|.*cancellation.*",
     ["Cancellations made 48 hours before check-in are free of charge. Late cancellations may incur a fee.",
      "Free cancellation is available up to 48 hours before your check-in date.",
      "You can cancel your booking for free if done 48 hours in advance. Please call us to cancel."]],

    [r".*pet.*|.*dog.*|.*cat.*|.*animal.*",
     ["We are a pet-friendly hotel! Pets are welcome with a small additional fee of $20 per night.",
      "Yes we welcome pets! There is a pet fee of $20/night. Please inform us in advance.",
      "Grand Palace Hotel is pet friendly. Pets are allowed with prior notice and a $20/night pet fee."]],

    [r".*parking.*|.*car.*",
     ["We offer free parking for all hotel guests. Valet parking is also available.",
      "Free parking is available on premises. Valet parking service is also available for your convenience.",
      "Yes we have free parking. Valet parking is available at $10/day if preferred."]],

    [r".*thank.*|.*thanks.*|.*thank you.*",
     ["You are most welcome! Is there anything else I can help you with?",
      "Happy to help! Let me know if you have any other questions.",
      "My pleasure! Enjoy your stay at Grand Palace Hotel!"]],

    [r"Bye|bye|Goodbye|goodbye|see you|exit",
     ["Goodbye! We hope to see you at Grand Palace Hotel soon!",
      "Thank you for choosing Grand Palace Hotel. Have a wonderful day!",
      "Bye! We look forward to welcoming you at Grand Palace Hotel!"]]
]
chatbot = Chat(pairs,reflections)
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat',methods=['POST'])
def chat():
    user_message=request.json.get('message')
    response = chatbot.respond(user_message.lower())
    if not response:
        response="I am sorry, I did not understand that. Could you please ask about rooms, prices, amenities, or booking?"
    return jsonify({'response':response})
if __name__=='__main__':
    app.run(debug=True)